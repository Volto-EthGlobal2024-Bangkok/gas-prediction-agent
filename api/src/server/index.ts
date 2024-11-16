import http from "http";
import {connectToDB} from "../libs/connection/connection";
import {loadAppConfiguration} from "../libs/env/env";

export async function startServer() {
    await loadAppConfiguration();

    await connectToDB();

    const serverImport = await import("./server");
    const server = await serverImport.default();
    const Server: http.Server = http.createServer(server);

    Server.listen(server.get("port"), () => {
        console.log(`listening on port:${server.get("port")}`);
    });
}

startServer();