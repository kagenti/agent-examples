FROM docker.io/library/golang:1.24.9-bookworm

WORKDIR /app
EXPOSE 8080

COPY . /app
RUN ls -l /app

RUN CGO_ENABLED=0 GOOS=linux go build -o /mcp-mitm /app/cli/main/main.go

CMD ["/mcp-mitm"]
