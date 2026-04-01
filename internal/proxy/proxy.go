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
	"sync/atomic"

	"github.com/tim/ollama-proxy/internal/config"
)

type Proxy struct {
	backends []config.Backend
	routes   map[string][]config.Backend       // model name → backends in priority order
	rp       map[string]*httputil.ReverseProxy  // backend name → reverse proxy
	healthy  map[string]*atomic.Bool            // backend name → health state
}

func New(cfg *config.Config) (*Proxy, error) {
	p := &Proxy{
		backends: cfg.Backends,
		routes:   make(map[string][]config.Backend),
		rp:       make(map[string]*httputil.ReverseProxy),
		healthy:  make(map[string]*atomic.Bool),
	}

	for _, b := range cfg.Backends {
		u, err := url.Parse(b.URL)
		if err != nil {
			return nil, fmt.Errorf("backend %q: invalid url %q: %w", b.Name, b.URL, err)
		}

		b := b // capture for closure
		rp := httputil.NewSingleHostReverseProxy(u)
		rp.ModifyResponse = stripOllamaVaryHeader
		rp.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			slog.Warn("backend transport error", "backend", b.Name, "error", err)
			p.markDown(b.Name)
			http.Error(w, "upstream unavailable", http.StatusBadGateway)
		}
		p.rp[b.Name] = rp

		h := &atomic.Bool{}
		h.Store(true) // assume healthy until first poll
		p.healthy[b.Name] = h

		// Explicit model list: append backend in config order for failover priority.
		for _, m := range b.Models {
			p.routes[m] = append(p.routes[m], b)
		}
	}

	p.discover()
	p.startHealthLoop()

	return p, nil
}

// discover queries each backend for its model list and populates the routing table.
// Backends are appended in config order so priority is preserved across all backends
// that carry the same model.
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
			// Append this backend only if not already listed (avoids duplicates
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
		// forward to the first healthy backend.
		b, ok := p.pickBackend(p.backends)
		if !ok {
			http.Error(w, "no healthy backends", http.StatusServiceUnavailable)
			return
		}
		p.forward(w, r, b)
	}
}

// handleChat routes POST /api/chat to the first healthy backend for the model.
func (p *Proxy) handleChat(w http.ResponseWriter, r *http.Request) {
	body, model, err := peekModel(r)
	if err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	backend, ok := p.resolveBackend(model)
	if !ok {
		http.Error(w, "no healthy backend for model: "+model, http.StatusServiceUnavailable)
		return
	}

	slog.Debug("routing chat", "model", model, "backend", backend.Name)
	r.Body = io.NopCloser(strings.NewReader(string(body)))

	if backend.Type == config.BackendOpenAI {
		p.forwardChatToOpenAI(w, r, body, backend)
		return
	}
	p.forward(w, r, backend)
}

// handleGenerate routes POST /api/generate to the first healthy backend for the model.
func (p *Proxy) handleGenerate(w http.ResponseWriter, r *http.Request) {
	body, model, err := peekModel(r)
	if err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	backend, ok := p.resolveBackend(model)
	if !ok {
		http.Error(w, "no healthy backend for model: "+model, http.StatusServiceUnavailable)
		return
	}

	slog.Debug("routing generate", "model", model, "backend", backend.Name)
	r.Body = io.NopCloser(strings.NewReader(string(body)))

	if backend.Type == config.BackendOpenAI {
		p.forwardGenerateToOpenAI(w, r, body, backend)
		return
	}
	p.forward(w, r, backend)
}

// handleShow routes POST /api/show to the first healthy backend for the model.
// Handles both {"model": "..."} (current API) and {"name": "..."} (legacy).
func (p *Proxy) handleShow(w http.ResponseWriter, r *http.Request) {
	body, model, err := peekModelOrName(r)
	if err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	backend, ok := p.resolveBackend(model)
	if !ok {
		http.Error(w, "no healthy backend for model: "+model, http.StatusServiceUnavailable)
		return
	}

	r.Body = io.NopCloser(strings.NewReader(string(body)))
	p.forward(w, r, backend)
}

// handleTags aggregates model lists from all healthy backends and merges them.
func (p *Proxy) handleTags(w http.ResponseWriter, r *http.Request) {
	type tagsResponse struct {
		Models []ollamaModel `json:"models"`
	}

	var mu sync.Mutex
	var all []ollamaModel

	var wg sync.WaitGroup
	for _, b := range p.backends {
		if !p.healthy[b.Name].Load() {
			continue
		}

		if b.Type == config.BackendOpenAI {
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

// resolveBackend returns the first healthy backend for the given model name,
// falling back to the first healthy backend overall if the model is unknown.
func (p *Proxy) resolveBackend(model string) (config.Backend, bool) {
	if backends, ok := p.routes[model]; ok {
		if b, ok := p.pickBackend(backends); ok {
			return b, true
		}
	}
	// Model unknown or all known backends down — try any healthy backend.
	slog.Warn("no healthy backend for model in routes, falling back", "model", model)
	return p.pickBackend(p.backends)
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
