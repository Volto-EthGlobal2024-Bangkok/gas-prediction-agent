import express from "express";
import * as Routes from "../routes";

export default async function setupServer() {
    const app: express.Application = express();

    Routes.init(app);

    /**
     * sets port 3000 to default or unless otherwise specified in the environment
     */
    app.set("port", process.env.PORT || 8080);

    return app;
}
