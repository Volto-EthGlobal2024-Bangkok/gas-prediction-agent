export interface GetGraphDataResponse {
    data: {
        day: [{
            values: [{
                date: string,
                values: number,
                node_number: number,
            }]
        }]
    },
}

export interface GasPrice {
    block_number: number,
    timestamp: Date,
    gas_price_gwei: number,
    day: string,
}