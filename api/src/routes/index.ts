import express, {NextFunction, Request, Response} from "express";

import PredictionRouter from "./PredictionRouter";
import GasRouter from "./GasRouter";

import cors from "cors";

export function init(app: express.Application): void {
    const router: express.Router = express.Router();

    const corsOptions = {
        origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
        credentials: true,
        optionsSuccessStatus: 200
    };


    app.use(cors(corsOptions));
    
    router.get(
        "/health",
        (req: Request, res: Response, next: NextFunction) => {
            /**
             * #swagger.tags = ['Healthcheck']
             * #swagger.description = 'Endpoint to check if the server is running'
             * #swagger.responses[200] = { description: 'Server is running' }
             */
            res.status(200).json({running: true});
        },
    );

    router.use("/api/v1/predict", PredictionRouter);

    router.use("/api/v1/gas", GasRouter);

    app.use(router);

/*     const corsOptions = {
        origin: (requestOrigin, callback) => {
            // Allow requests with no origin (like mobile apps or curl requests)
            if (!requestOrigin) return callback(null, true);
    
            // Allow localhost with any protocol (http/https) and any port
            if (requestOrigin.match(/^http(s)?:\/\/localhost:\d+$/)) {
                callback(null, true);
            } else {
                callback(new Error("Not allowed by CORS"));
            }
        },
        methods: "GET,HEAD,PUT,PATCH,POST,DELETE",
        credentials: true,
    };*/


}
