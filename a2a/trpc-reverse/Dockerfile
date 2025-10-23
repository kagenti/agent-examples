FROM golang:1.24.9-bookworm

WORKDIR /app
EXPOSE 8080

COPY main.go go.mod go.sum /app
RUN ls -l /app

RUN CGO_ENABLED=0 GOOS=linux go build -o /trpc-reverse main.go

CMD ["/trpc-reverse", "-host", "0.0.0.0", "-port", "8080"]
