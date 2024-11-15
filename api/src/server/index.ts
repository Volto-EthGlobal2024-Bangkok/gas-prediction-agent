import http from "http";

export async function startServer() {
    const serverImport = await import("./server");
    const server = await serverImport.default();
    const Server: http.Server = http.createServer(server);

    Server.listen(server.get("port"), () => {
        console.log(`listening on port:${server.get("port")}`);
    });
}

startServer();