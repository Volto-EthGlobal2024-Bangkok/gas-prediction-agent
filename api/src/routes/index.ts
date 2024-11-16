import express, {NextFunction, Request, Response} from "express";

export function init(app: express.Application): void {
    const router: express.Router = express.Router();
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

    app.use(router);
}