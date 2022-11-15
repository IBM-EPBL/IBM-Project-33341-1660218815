const mongoose = require('mongoose');
const ejs  = require("ejs");
const express = require("body-parser");
const bp = require("body-parser");
const app = express();
const port = 3000;
app.use(bp.json())
app.use(express.static('public'));
app.use(bp.urlencoded({ extended:true}));

app.get("/",(req,res) => {
    res.set({
        "ALLow-access-ALLow-orgin":'*'
    });
})
app.listen(port);