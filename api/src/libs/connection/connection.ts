import * as mongoose from "mongoose";
import {getAppConfig} from "../env/env";

let DB: mongoose.Connection;

export async function connectToDB() {
    const config = getAppConfig();
    console.log(config.mongo.MONGODB_URI);
    const MONGO_URI: string = "mongodb://localhost:27017/productiondb?retryWrites=true&w=majority";

    DB = mongoose.createConnection(MONGO_URI);

    // handlers
    DB.on("connecting", () => {
        console.log(`\x1b[32mMongoDB :: connecting`);
    });

    DB.on("error", (error: any) => {
        console.log(`\x1b[31mMongoDB :: connection ${error}`);
        mongoose.disconnect();
    });

    DB.on("connected", () => {
        console.log(`\x1b[32mMongoDB :: connecting`);
    });

    DB.once("open", () => {
        console.log(`\x1b[32mMongoDB :: connection opened`);
    });

    DB.on("reconnected", () => {
        console.log(`\x1b[32mMongoDB :: reconnected`);
    });

    DB.on("reconnectFailed", () => {
        console.log(`\x1b[31mMongoDB :: reconnectFailed`);
    });

    DB.on("disconnected", () => {
        console.log(`\x1b[31mMongoDB :: disconnected`);
    });

    DB.on("fullsetup", () => {
        console.log(`\x1b[33mMongoDB :: reconnecting... %d`);
    });

    return DB;
}

export function getConnection() {
    return DB;
}
