import * as dotenv from "dotenv";

dotenv.config();

interface IConfig {
    mongo: {
        MONGODB_URI: string;
    }
}

let config: {
    [name: string]: IConfig;
} = {};

export const NODE_ENV: string = process.env.NODE_ENV || "production";

export async function loadAppConfiguration() {
    config[NODE_ENV] = {
        mongo: {
            MONGODB_URI: process.env.MONGODB_URI || "random",//"mongodb://localhost:27017/productiondb?retryWrites=true&w=majority",
        },
    };

    return config;
}

export function getAppConfig() {
    return config[NODE_ENV];
}
