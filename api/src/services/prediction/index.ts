import {NextFunction, Request, Response} from "express";
import PredictionService from "./service";

export async function predict(
    req: Request,
    res: Response,
    next: NextFunction,
) {
    try {
        const days = parseInt(req.params.days);
        const prediction = await PredictionService.predict(days);
        res.status(200).json(prediction);
    } catch (error) {
        console.log(
            `Error found in ${__filename} - getFAQs method: ${error.message}`,
        );
        next(error);
    }
}
