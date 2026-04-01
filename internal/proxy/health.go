package proxy

import (
	"context"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/tim/ollama-proxy/internal/config"
)

const (
	healthPollInterval = 15 * time.Second
	healthCheckTimeout = 5 * time.Second
)

// startHealthLoop runs an initial health check then polls every 15s in the background.
func (p *Proxy) startHealthLoop() {
	p.pollAll()
	go func() {
		t := time.NewTicker(healthPollInterval)
		defer t.Stop()
		for range t.C {
			p.pollAll()
		}
	}()
}

// pollAll checks all backends concurrently.
func (p *Proxy) pollAll() {
	for _, b := range p.backends {
		b := b
		go func() {
			ok := p.ping(b)
			prev := p.healthy[b.Name].Swap(ok)
			if prev != ok {
				if ok {
					slog.Info("backend recovered", "backend", b.Name)
				} else {
					slog.Warn("backend down", "backend", b.Name)
				}
			}
		}()
	}
}

// ping returns true if the backend responds to a health endpoint within 5s.
func (p *Proxy) ping(b config.Backend) bool {
	ctx, cancel := context.WithTimeout(context.Background(), healthCheckTimeout)
	defer cancel()

	var u string
	if b.Type == config.BackendOpenAI {
		u = strings.TrimRight(b.URL, "/") + "/models"
	} else {
		u = strings.TrimRight(b.URL, "/") + "/api/tags"
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return false
	}
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode < 500
}

// markDown immediately marks a backend unhealthy without waiting for the next poll.
func (p *Proxy) markDown(name string) {
	if h, ok := p.healthy[name]; ok {
		if h.Swap(false) {
			slog.Warn("backend marked down on request failure", "backend", name)
		}
	}
}

// pickBackend returns the first healthy backend from the list.
func (p *Proxy) pickBackend(backends []config.Backend) (config.Backend, bool) {
	for _, b := range backends {
		if p.healthy[b.Name].Load() {
			return b, true
		}
	}
	return config.Backend{}, false
}

// healthyBool is an alias so proxy.go can reference the type without importing sync/atomic.
type healthyBool = atomic.Bool
