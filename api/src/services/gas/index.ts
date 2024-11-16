import {NextFunction, Request, Response} from "express";
import PredictionService from "./service";
import GraphService from "./service";

export async function getData(
    req: Request,
    res: Response,
    next: NextFunction,
) {
    try {
        const data = await GraphService.getData();
        res.status(200).json(data);
    } catch (error) {
        console.log(
            `Error found in ${__filename} - getFAQs method: ${error.message}`,
        );
        next(error);
    }
}
