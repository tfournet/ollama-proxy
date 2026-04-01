package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/tim/ollama-proxy/internal/config"
)

// Ollama /api/chat request shape.
type ollamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   *bool               `json:"stream,omitempty"`
	Options  map[string]any      `json:"options,omitempty"`
}

type ollamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Ollama /api/generate request shape.
type ollamaGenerateRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	System  string         `json:"system,omitempty"`
	Stream  *bool          `json:"stream,omitempty"`
	Options map[string]any `json:"options,omitempty"`
}

// OpenAI request/response shapes (minimal).
type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []ollamaChatMessage `json:"messages"`
	Stream      bool                `json:"stream"`
	Temperature *float64            `json:"temperature,omitempty"`
	MaxTokens   *int                `json:"max_tokens,omitempty"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message      *ollamaChatMessage `json:"message,omitempty"`
		Delta        *ollamaChatMessage `json:"delta,omitempty"`
		FinishReason *string            `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage,omitempty"`
}

type openAIModel struct {
	ID string `json:"id"`
}

type openAIModelsResponse struct {
	Data []openAIModel `json:"data"`
}

// forwardChatToOpenAI translates an Ollama /api/chat request to an OpenAI
// /v1/chat/completions request, forwards it, and translates the response back.
func (p *Proxy) forwardChatToOpenAI(w http.ResponseWriter, r *http.Request, body []byte, b config.Backend) {
	var req ollamaChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "parse request: "+err.Error(), http.StatusBadRequest)
		return
	}

	stream := req.Stream == nil || *req.Stream

	oaiReq := openAIChatRequest{
		Model:    req.Model,
		Messages: req.Messages,
		Stream:   stream,
	}
	if t, ok := req.Options["temperature"].(float64); ok {
		oaiReq.Temperature = &t
	}
	if n, ok := req.Options["num_predict"].(float64); ok {
		v := int(n)
		oaiReq.MaxTokens = &v
	}

	reqBody, err := json.Marshal(oaiReq)
	if err != nil {
		http.Error(w, "marshal request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	oaiURL := strings.TrimRight(b.URL, "/") + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, oaiURL, bytes.NewReader(reqBody))
	if err != nil {
		http.Error(w, "create request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if b.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+b.APIKey)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		p.markDown(b.Name)
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}

	now := time.Now().UTC().Format(time.RFC3339Nano)

	if stream {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Transfer-Encoding", "chunked")

		flusher, _ := w.(http.Flusher)
		scanner := bufio.NewScanner(resp.Body)

		for scanner.Scan() {
			line := strings.TrimPrefix(scanner.Text(), "data: ")
			if line == "" || line == "[DONE]" {
				continue
			}

			var chunk openAIChatResponse
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				slog.Warn("parse openai chunk", "error", err, "line", line)
				continue
			}
			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			done := choice.FinishReason != nil && *choice.FinishReason != ""
			content := ""
			if choice.Delta != nil {
				content = choice.Delta.Content
			}

			ollamaChunk := map[string]any{
				"model":      req.Model,
				"created_at": now,
				"message": map[string]string{
					"role":    "assistant",
					"content": content,
				},
				"done": done,
			}

			out, _ := json.Marshal(ollamaChunk)
			fmt.Fprintf(w, "%s\n", out)
			if flusher != nil {
				flusher.Flush()
			}
		}

		// Final done message.
		final := map[string]any{
			"model":      req.Model,
			"created_at": now,
			"message":    map[string]string{"role": "assistant", "content": ""},
			"done":       true,
		}
		out, _ := json.Marshal(final)
		fmt.Fprintf(w, "%s\n", out)
		if flusher != nil {
			flusher.Flush()
		}

		return
	}

	// Non-streaming response.
	var oaiResp openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaiResp); err != nil {
		http.Error(w, "decode response: "+err.Error(), http.StatusBadGateway)
		return
	}

	content := ""
	if len(oaiResp.Choices) > 0 && oaiResp.Choices[0].Message != nil {
		content = oaiResp.Choices[0].Message.Content
	}

	ollamaResp := map[string]any{
		"model":      req.Model,
		"created_at": now,
		"message": map[string]string{
			"role":    "assistant",
			"content": content,
		},
		"done": true,
	}
	if oaiResp.Usage != nil {
		ollamaResp["prompt_eval_count"] = oaiResp.Usage.PromptTokens
		ollamaResp["eval_count"] = oaiResp.Usage.CompletionTokens
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ollamaResp)
}

// forwardGenerateToOpenAI converts /api/generate to a single-turn chat request.
func (p *Proxy) forwardGenerateToOpenAI(w http.ResponseWriter, r *http.Request, body []byte, b config.Backend) {
	var req ollamaGenerateRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "parse request: "+err.Error(), http.StatusBadRequest)
		return
	}

	messages := []ollamaChatMessage{}
	if req.System != "" {
		messages = append(messages, ollamaChatMessage{Role: "system", Content: req.System})
	}
	messages = append(messages, ollamaChatMessage{Role: "user", Content: req.Prompt})

	stream := req.Stream == nil || *req.Stream
	chatBody, _ := json.Marshal(ollamaChatRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   &stream,
		Options:  req.Options,
	})

	r.Body = io.NopCloser(bytes.NewReader(chatBody))
	p.forwardChatToOpenAI(w, r, chatBody, b)
}

// fetchOpenAIModels returns the model list from an OpenAI backend in Ollama format.
func (p *Proxy) fetchOpenAIModels(b config.Backend) []ollamaModel {
	modelsURL := strings.TrimRight(b.URL, "/") + "/models"
	req, err := http.NewRequest(http.MethodGet, modelsURL, nil) //nolint:noctx
	if err != nil {
		slog.Warn("openai models request", "backend", b.Name, "error", err)
		return nil
	}
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Warn("openai models fetch", "backend", b.Name, "error", err)
		return nil
	}
	defer resp.Body.Close()

	var oaiResp openAIModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaiResp); err != nil {
		slog.Warn("openai models decode", "backend", b.Name, "error", err)
		return nil
	}

	now := time.Now().UTC().Format(time.RFC3339Nano)
	models := make([]ollamaModel, 0, len(oaiResp.Data))
	for _, m := range oaiResp.Data {
		models = append(models, ollamaModel{
			Name:       m.ID,
			Model:      m.ID,
			ModifiedAt: now,
		})
	}
	return models
}
