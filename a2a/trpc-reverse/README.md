
# trpc-reverse

This is a copy of [a trpc-example](https://github.com/trpc-group/trpc-a2a-go/blob/main/examples/simple/README.md) that reverses text.

To run, `go run main.go -host 0.0.0.0 -port 9090`

To build a container, `docker build --load --tag trpc-reverse:v0.0.1 .`

To load onto a Kind cluster, such as the Kagenti test environment, `kind load -n agent-platform docker-image trpc-reverse:v0.0.1`

