import mongoose, { Model, Schema } from "mongoose";
import {GasPrice} from "./interfaces";

const GasPriceSchema: Schema = new Schema<GasPrice>({
    block_number: {type: Number},
    timestamp: {type: Date},
    gas_price_gwei: {type: Number},
    day: {type: String},
});

export let GasPriceModel: Model<GasPrice> = null;

export function getGasPriceModel(DB: mongoose.Connection): Model<GasPrice> {
    if (!GasPriceModel) {
        GasPriceModel = DB.model<GasPrice>(
            "gasprice",
            GasPriceSchema,
        );
    }
    return GasPriceModel;
}