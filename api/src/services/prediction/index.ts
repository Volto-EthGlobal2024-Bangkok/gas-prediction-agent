import {NextFunction, Request, Response} from "express";
import PredictionService from "./service";

export async function predict(
    req: Request,
    res: Response,
    next: NextFunction,
) {
    try {
        const days = parseInt(req.params.days);
        const faqs = await PredictionService.predict(days);
        res.status(200).json(faqs);
    } catch (error) {
        console.log(
            `Error found in ${__filename} - getFAQs method: ${error.message}`,
        );
        next(error);
    }
}