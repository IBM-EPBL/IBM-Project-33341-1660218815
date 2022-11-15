const mongoose=require("mongoose");
const ejs=require("ejs");
const express=require("express");
const bp = require("body-parser");
const app = express();

const db="mongodb+srv://<jayaram>:<jayaramdharani>@cluster0.bpvhs.mongodb.net/database?retryWrites=true&w=majority";
const connectp={
    useNewUrlParser:true,
    useUnifiedTopology: true
};
mongoose.connect(db,connectp).then(()=>{console.info("connected to the DB");})
.catch((e)=>{
    console.log("error:",e);
});