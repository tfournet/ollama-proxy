package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"

	"github.com/tim/ollama-proxy/internal/config"
)

type Proxy struct {
	backends []config.Backend
	routes   map[string][]config.Backend      // model name → backends in priority order
	rp       map[string]*httputil.ReverseProxy // backend name → reverse proxy
}

func New(cfg *config.Config) (*Proxy, error) {
	p := &Proxy{
		backends: cfg.Backends,
		routes:   make(map[string][]config.Backend),
		rp:       make(map[string]*httputil.ReverseProxy),
	}

	for _, b := range cfg.Backends {
		u, err := url.Parse(b.URL)
		if err != nil {
			return nil, fmt.Errorf("backend %q: invalid url %q: %w", b.Name, b.URL, err)
		}
		rp := httputil.NewSingleHostReverseProxy(u)
		rp.ModifyResponse = stripOllamaVaryHeader
		p.rp[b.Name] = rp

		// Explicit model list: append backend in config order for failover.
		for _, m := range b.Models {
			p.routes[m] = append(p.routes[m], b)
		}
	}

	// Auto-discover models from each backend. Explicit entries above take
	// priority; discovered models only fill in gaps.
	p.discover()

	return p, nil
}

// discover queries each backend for its model list and registers any models
// not already in the routing table.
func (p *Proxy) discover() {
	for _, b := range p.backends {
		var names []string

		if b.Type == config.BackendOpenAI {
			for _, m := range p.fetchOpenAIModels(b) {
				names = append(names, m.Name)
			}
		} else {
			tagsURL := strings.TrimRight(b.URL, "/") + "/api/tags"
			resp, err := http.Get(tagsURL) //nolint:noctx
			if err != nil {
				slog.Warn("model discovery failed", "backend", b.Name, "error", err)
				continue
			}
			var result struct {
				Models []struct {
					Name string `json:"name"`
				} `json:"models"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				resp.Body.Close()
				slog.Warn("model discovery decode failed", "backend", b.Name, "error", err)
				continue
			}
			resp.Body.Close()
			for _, m := range result.Models {
				names = append(names, m.Name)
			}
		}

		added := 0
		for _, name := range names {
			// Append this backend only if not already listed (avoid duplicates
			// when explicit model list and auto-discovery overlap).
			already := false
			for _, existing := range p.routes[name] {
				if existing.Name == b.Name {
					already = true
					break
				}
			}
			if !already {
				p.routes[name] = append(p.routes[name], b)
				added++
			}
		}
		slog.Info("models discovered", "backend", b.Name, "added", added, "total", len(names))
	}
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	slog.Debug("request", "method", r.Method, "path", r.URL.Path)

	switch {
	case r.URL.Path == "/api/tags" && r.Method == http.MethodGet:
		p.handleTags(w, r)
	case r.URL.Path == "/api/chat" && r.Method == http.MethodPost:
		p.handleChat(w, r)
	case r.URL.Path == "/api/generate" && r.Method == http.MethodPost:
		p.handleGenerate(w, r)
	case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
		p.handleShow(w, r)
	default:
		// For all other endpoints (pull, push, delete, embed, etc.),
		// forward to the first backend.
		if len(p.backends) == 0 {
			http.Error(w, "no backends configured", http.StatusServiceUnavailable)
			return
		}
		p.forward(w, r, p.backends[0])
	}
}

// handleChat routes POST /api/chat based on the model field.
func (p *Proxy) handleChat(w http.ResponseWriter, r *http.Request) {
	body, model, err := peekModel(r)
	if err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	backends := p.routes[model]
	if len(backends) == 0 {
		slog.Warn("unknown model, using first backend", "model", model)
		if len(p.backends) == 0 {
			http.Error(w, "no backends configured", http.StatusServiceUnavailable)
			return
		}
		backends = []config.Backend{p.backends[0]}
	}

	p.tryChatBackends(w, r, body, model, backends)
}

// tryChatBackends attempts each backend in order, falling back on error or
// model-not-found responses.
func (p *Proxy) tryChatBackends(w http.ResponseWriter, r *http.Request, body []byte, model string, backends []config.Backend) {
	for i, backend := range backends {
		last := i == len(backends)-1
		slog.Debug("routing chat", "model", model, "backend", backend.Name, "attempt", i+1)

		if backend.Type == config.BackendOpenAI {
			rec := newResponseRecorder()
			p.forwardChatToOpenAI(rec, r, body, backend)
			if rec.code < 500 && rec.code != http.StatusNotFound {
				rec.writeTo(w)
				return
			}
			if !last {
				slog.Warn("chat failover", "model", model, "from", backend.Name, "status", rec.code)
				continue
			}
			rec.writeTo(w)
			return
		}

		if last {
			r.Body = io.NopCloser(strings.NewReader(string(body)))
			p.forward(w, r, backend)
			return
		}

		rec := newResponseRecorder()
		r.Body = io.NopCloser(strings.NewReader(string(body)))
		p.forwardTo(rec, r, backend)
		if rec.code < 500 && rec.code != http.StatusNotFound {
			rec.writeTo(w)
			return
		}
		slog.Warn("chat failover", "model", model, "from", backend.Name, "status", rec.code)
	}
}

// handleGenerate routes POST /api/generate based on the model field.
func (p *Proxy) handleGenerate(w http.ResponseWriter, r *http.Request) {
	body, model, err := peekModel(r)
	if err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	backends := p.routes[model]
	if len(backends) == 0 {
		slog.Warn("unknown model, using first backend", "model", model)
		if len(p.backends) == 0 {
			http.Error(w, "no backends configured", http.StatusServiceUnavailable)
			return
		}
		backends = []config.Backend{p.backends[0]}
	}

	for i, backend := range backends {
		last := i == len(backends)-1
		slog.Debug("routing generate", "model", model, "backend", backend.Name, "attempt", i+1)

		if backend.Type == config.BackendOpenAI {
			if last {
				p.forwardGenerateToOpenAI(w, r, body, backend)
				return
			}
			rec := newResponseRecorder()
			p.forwardGenerateToOpenAI(rec, r, body, backend)
			if rec.code < 500 && rec.code != http.StatusNotFound {
				rec.writeTo(w)
				return
			}
			slog.Warn("generate failover", "model", model, "from", backend.Name, "status", rec.code)
			continue
		}

		if last {
			r.Body = io.NopCloser(strings.NewReader(string(body)))
			p.forward(w, r, backend)
			return
		}

		rec := newResponseRecorder()
		r.Body = io.NopCloser(strings.NewReader(string(body)))
		p.forwardTo(rec, r, backend)
		if rec.code < 500 && rec.code != http.StatusNotFound {
			rec.writeTo(w)
			return
		}
		slog.Warn("generate failover", "model", model, "from", backend.Name, "status", rec.code)
	}
}

// handleShow routes POST /api/show based on the model field.
// Handles both {"model": "..."} (current API) and {"name": "..."} (legacy).
func (p *Proxy) handleShow(w http.ResponseWriter, r *http.Request) {
	body, model, err := peekModelOrName(r)
	if err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	backends := p.routes[model]
	var backend config.Backend
	if len(backends) > 0 {
		backend = backends[0]
	} else if len(p.backends) > 0 {
		backend = p.backends[0]
	} else {
		http.Error(w, "no backends configured", http.StatusServiceUnavailable)
		return
	}

	r.Body = io.NopCloser(strings.NewReader(string(body)))
	p.forward(w, r, backend)
}

// handleTags aggregates model lists from all backends and merges them.
func (p *Proxy) handleTags(w http.ResponseWriter, r *http.Request) {
	type tagsResponse struct {
		Models []ollamaModel `json:"models"`
	}

	var mu sync.Mutex
	var all []ollamaModel

	var wg sync.WaitGroup
	for _, b := range p.backends {
		if b.Type == config.BackendOpenAI {
			// Convert OpenAI model list to Ollama format.
			wg.Add(1)
			go func(b config.Backend) {
				defer wg.Done()
				models := p.fetchOpenAIModels(b)
				mu.Lock()
				all = append(all, models...)
				mu.Unlock()
			}(b)
			continue
		}

		wg.Add(1)
		go func(b config.Backend) {
			defer wg.Done()
			tagsURL := strings.TrimRight(b.URL, "/") + "/api/tags"
			resp, err := http.Get(tagsURL) //nolint:noctx
			if err != nil {
				slog.Warn("tags fetch failed", "backend", b.Name, "error", err)
				return
			}
			defer resp.Body.Close()

			var tr tagsResponse
			if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
				slog.Warn("tags decode failed", "backend", b.Name, "error", err)
				return
			}

			mu.Lock()
			all = append(all, tr.Models...)
			mu.Unlock()
		}(b)
	}
	wg.Wait()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tagsResponse{Models: all})
}

// forward proxies the request to the given backend unchanged.
func (p *Proxy) forward(w http.ResponseWriter, r *http.Request, b config.Backend) {
	rp, ok := p.rp[b.Name]
	if !ok {
		http.Error(w, "backend not found: "+b.Name, http.StatusInternalServerError)
		return
	}
	rp.ServeHTTP(w, r)
}

// forwardTo proxies to backend into a ResponseWriter (may be a recorder).
func (p *Proxy) forwardTo(w http.ResponseWriter, r *http.Request, b config.Backend) {
	p.forward(w, r, b)
}

// responseRecorder captures a response so we can inspect it before writing to
// the real ResponseWriter (needed for failover decisions).
type responseRecorder struct {
	code    int
	headers http.Header
	buf     strings.Builder
}

func newResponseRecorder() *responseRecorder {
	return &responseRecorder{code: http.StatusOK, headers: make(http.Header)}
}

func (r *responseRecorder) Header() http.Header         { return r.headers }
func (r *responseRecorder) WriteHeader(code int)        { r.code = code }
func (r *responseRecorder) Write(b []byte) (int, error) { return r.buf.Write(b) }

func (r *responseRecorder) writeTo(w http.ResponseWriter) {
	for k, vv := range r.headers {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(r.code)
	io.WriteString(w, r.buf.String()) //nolint:errcheck
}

// peekModel reads the request body, extracts the "model" field, and returns
// both the raw body bytes and the model name.
func peekModel(r *http.Request) ([]byte, string, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, "", fmt.Errorf("read body: %w", err)
	}

	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, "", fmt.Errorf("parse model field: %w", err)
	}
	if payload.Model == "" {
		return nil, "", fmt.Errorf("model field is empty")
	}

	return body, payload.Model, nil
}

// peekModelOrName is like peekModel but also accepts the legacy "name" field
// used by older Ollama clients (e.g. Open WebUI's /api/show calls).
func peekModelOrName(r *http.Request) ([]byte, string, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, "", fmt.Errorf("read body: %w", err)
	}

	var payload struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, "", fmt.Errorf("parse body: %w", err)
	}

	model := payload.Model
	if model == "" {
		model = payload.Name
	}
	if model == "" {
		return nil, "", fmt.Errorf("model field is empty")
	}

	return body, model, nil
}

func stripOllamaVaryHeader(resp *http.Response) error {
	resp.Header.Del("Vary")
	return nil
}
