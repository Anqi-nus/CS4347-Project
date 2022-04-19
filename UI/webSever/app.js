var express = require('express');
var app = express();
var path = require('path');
var fs = require('fs')
var multer = require('multer')
//设置跨域访问
app.all("*", function (req, res, next) {
    //设置允许跨域的域名，*代表允许任意域名跨域
    res.header("Access-Control-Allow-Origin", req.headers.origin || '*');
    // //允许的header类型
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
    // //跨域允许的请求方式
    res.header("Access-Control-Allow-Methods", "PUT,POST,GET,DELETE,OPTIONS");
    // 可以带cookies
    res.header("Access-Control-Allow-Credentials", true);
    if (req.method == 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
})


app.use(express.static(path.join(__dirname, 'public')));
//模板引擎
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.get("/", (req, res, next) => {
    res.render("index")
})
//上传文件
app.post('/upload', (req, res, next) => {

    var upload = multer({ dest: 'upload/' }).any();

    upload(req, res, err => {
        if (err) {
            console.log(err);
            return
        }
        let file = req.files[0]
        let filname = file.originalname
        var oldPath = file.path
        var newPath = path.join(process.cwd(), "upload/" + new Date().getTime()+filname)
        var src = fs.createReadStream(oldPath);
        var dest = fs.createWriteStream(newPath);
        src.pipe(dest);
        src.on("end", () => {
            let filepath = path.join(process.cwd(), oldPath)
            fs.unlink(filepath, err => {
                if (err) {
                    console.log(err);
                    return
                }
                res.send("ok")
            })

        })
        src.on("error", err => {
            res.end("err")
        })

    })

})


app.use((req, res) => {
    res.send("404")
})
app.listen(5000)
