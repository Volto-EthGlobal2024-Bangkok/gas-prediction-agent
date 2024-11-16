import * as PredictionComponent from "./prediction";
import * as GasComponent from "./gas";
import mongoose from "mongoose";
import {getGasPriceModel} from "./gas/model";

export { PredictionComponent, GasComponent };

export async function runModels(DB: mongoose.Connection) {
    // Gas Price
    getGasPriceModel(DB);

    console.log("Models are running");
}