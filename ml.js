const ExcelJS = require('exceljs');
const tf = require('@tensorflow/tfjs');
const xlsFile = require('read-excel-file/node')
const fmincg = require('./fmincg')
//const tfvis = require('@tensorflow/tfjs-vis');

async function getData() {
    try{
        const workbookY = new ExcelJS.Workbook();
        const workbookR = new ExcelJS.Workbook();
        const workbookX = new ExcelJS.Workbook();
        const workbookTheta = new ExcelJS.Workbook();
        await workbookY.xlsx.readFile('./Y.xlsx');
        await workbookR.xlsx.readFile('./R.xlsx');
        await workbookX.xlsx.readFile('./X.xlsx');
        await workbookTheta.xlsx.readFile('./Theta.xlsx');
        const worksheetY = workbookY.getWorksheet(1);
        const worksheetR = workbookR.getWorksheet(1);
        const worksheetX = workbookX.getWorksheet(1);
        const worksheetTheta = workbookTheta.getWorksheet(1);
        let Y = []
        let R = []
        let X = []
        let Theta = []
        let nm = worksheetY.getColumn(1).values.length-1
        let nu = worksheetY.getRow(1).values.length-1
        let nf = worksheetX.getRow(1).values.length-1

        for(let i = 1; i <= nm ; i++){
            let Yrow = worksheetY.getRow(i).values
            Yrow.shift()
            Y.push(Yrow)
            let Rrow = worksheetR.getRow(i).values
            Rrow.shift()
            R.push(Rrow)
            let Xrow = worksheetX.getRow(i).values
            Xrow.shift()
            X.push(Xrow)
        }

        for(let i = 1; i <= nu ; i++){
            let Thetarow = worksheetTheta.getRow(i).values
            Thetarow.shift()
            Theta.push(Thetarow)
        }

        //nm = 25
        //nu = 10
        //return [nm, nu, nf, tf.tensor2d(Y).slice([0, 0],[25, 10]), tf.tensor2d(R).slice([0, 0],[25, 10]), tf.tensor2d(X).slice([0, 0],[25, 10]), tf.tensor2d(Theta).slice([0, 0],[10, 10])]
        return [nm, nu, nf, tf.tensor2d(Y), tf.tensor2d(R), tf.tensor2d(X), tf.tensor2d(Theta)]  
    }catch(err){
        console.log(err)
    }    
}

async function cosiCostFunc(params, Y, R, nu, nm, nf, lambda){
    try{
        let X = params.slice([0],[nm*nf]).reshape([nm, nf])
        let Theta = params.slice([nm*nf]).reshape([nu, nf])
        //X.slice([0, 0],[5, 3]).print()
        //Theta.slice([0, 0],[4, 3]).print()
        let J = tf.scalar(0)
        let X_grad = tf.zerosLike(X)
        let Theta_grad = tf.zerosLike(Theta)

        let temp = tf.sub(tf.matMul(X, Theta, false, true), Y)
        //temp.print()
        let reg1 = tf.sum(tf.pow(Theta, 2))
        //reg1.print()
        let reg2 = tf.sum(tf.pow(X, 2))
        //reg2.print()
        J = tf.div(tf.sum(tf.mul(tf.pow(temp, 2), R)),2)
        let reg = tf.mul(tf.div(lambda, 2), tf.add(reg1, reg2))
        J = tf.add(J, reg)
        X_grad = tf.add(tf.matMul(tf.mul(temp, R), Theta), tf.mul(lambda, X))
        //X_grad.slice([0, 0],[5, 3]).print()
        Theta_grad = tf.add(tf.matMul(tf.mul(temp, R), X, true, false), tf.mul(lambda, Theta))
        //Theta_grad.slice([0, 0],[4, 3]).print()
        let grad = tf.concat([X_grad, Theta_grad]).reshape([nu*nf+nm*nf, 1])
        //J.print()
        return [J, grad]
    }catch(err){
        console.log(err)
    }    
}

function normalizeRatings(nm, nu, Y, R){
    let Ysum = tf.sum(Y, 1).reshape([nm, 1])
    let Rsum = tf.sum(R, 1).reshape([nm, 1])
    let Ymean = tf.div(Ysum, Rsum).reshape([nm, 1])
    let Ysub = tf.mul(R, Ymean)
    let Ynorm = tf.sub(Y, Ysub)
    return [Ynorm, Ymean]
}

async function run(){
    let [nm, nu, nf, Y, R, X, Theta] = await getData()
    let [Ynorm, Ymean] = normalizeRatings(nm, nu, Y, R)
    //nu = 4
    //nm = 5
    //nf = 3
    X = tf.randomNormal([nm, nf])
    Theta = tf.randomNormal([nu, nf])
    // X = tf.tensor2d([1.048685501766515, -0.400231959974807, 1.194119448400868,
    //                     0.7808512309967017, -0.3856259111501408, 0.5211977859038951,
    //                     0.6415088601737823, -0.5478538451937864, -0.08379637792917273,
    //                     0.4536178246455854, -0.8002184375679769, 0.6804812850198093,
    //                     0.9375378888571004, 0.106089899023496, 0.361952953161411], [nm, nf]) 
    // Theta = tf.tensor2d([0.2854436154321042, -1.684265085744985, 0.2629387682498409,
    //                         0.5050132141977508, -0.4546484610447612, 0.3174624418572172,
    //                         -0.4319165610096791, -0.4788044894643126, 0.8467111057626113, 
    //                         0.7285983851255382, -0.2718939052901861, 0.3268436042370519],[nu, nf])
    let lambda = 10
    let params = tf.concat([X, Theta]).reshape([nu*nf+nm*nf, 1])
    //params.slice([0],[nm*nf]).print()
    //let arr = await cosiCostFunc(params, tf.slice(Y, [0, 0],[nm, nu]), tf.slice(R, [0, 0],[nm, nu]), nu, nm, nf, 0)
    //let arr = await cosiCostFunc(params, tf.slice(Y, [0, 0],[nm, nu]), tf.slice(R, [0, 0],[nm, nu]), nu, nm, nf, 1.5)
    // let arr = await cosiCostFunc(params, Ynorm, R, nu, nm, nf, lambda)
    //let J = arr[0]
    //let grad = arr[1] 
    //console.log(J.arraySync())  
    // for(let i = 0; i < 10; i++){
    //     params = tf.sub(params, tf.mul(tf.scalar(0.1), grad))
    //     //arr = await cosiCostFunc(params, tf.slice(Y, [0, 0],[nm, nu]), tf.slice(R, [0, 0],[nm, nu]), nu, nm, nf, 1.5)        
    //     arr = await cosiCostFunc(params, Ynorm, R, nu, nm, nf, lambda)
    //     J = arr[0]
    //     grad = arr[1]
    // }
    // grad.print()
    //let learningRate = 1.5
    //let optimizer = tf.train.adam(learningRate)
    //optimizer.minimize()
    const options = {maxIter: 3}
    let arguments = {Ynorm, R, nu, nm, nf, lambda}
    let [newParams, costHostory, iteration] = await fmincg(cosiCostFunc, arguments, params, options)
    X = newParams.slice([0],[nm*nf]).reshape([nm, nf])
    Theta = newParams.slice([nm*nf]).reshape([nu, nf])
    let predictions = tf.add(tf.matMul(X, Theta, false, true), Ymean)
    predictions.print()
}   

run()