package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type BackendType string

const (
	BackendOllama BackendType = "ollama"
	BackendOpenAI BackendType = "openai"
)

type Backend struct {
	Name   string      `yaml:"name"`
	Type   BackendType `yaml:"type"`
	URL    string      `yaml:"url"`
	APIKey string      `yaml:"api_key"`
	Models []string    `yaml:"models"`
}

type Config struct {
	Listen   string    `yaml:"listen"`
	Backends []Backend `yaml:"backends"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if cfg.Listen == "" {
		cfg.Listen = ":11435"
	}

	for i, b := range cfg.Backends {
		if b.Type == "" {
			cfg.Backends[i].Type = BackendOllama
		}
		if b.URL == "" {
			return nil, fmt.Errorf("backend %q has no url", b.Name)
		}
	}

	return &cfg, nil
}
