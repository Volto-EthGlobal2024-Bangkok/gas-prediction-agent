import {Router} from "express";
import {GasComponent} from "../services";

const router: Router = Router();

router.get("/data", GasComponent.getData);

export default router;