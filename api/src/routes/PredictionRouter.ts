import { Router } from "express";
import {PredictionComponent} from "../services";

const router: Router = Router();

router.get("/:days", PredictionComponent.predict);

export default router;