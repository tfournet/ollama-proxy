FROM alpine:3.19
COPY ollama-proxy-static /ollama-proxy
COPY servers.yaml /servers.yaml
EXPOSE 11435
ENTRYPOINT ["/ollama-proxy", "--config", "/servers.yaml"]
