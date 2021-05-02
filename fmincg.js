const tf = require('@tensorflow/tfjs');

async function fmincg(f, arguments, X, options){
    //f async function [J, grad]
    //X initial values
    //options options = optimset('GradObj', 'on', 'MaxIter', 10);
    let length = 100
    if(options && options.maxIter){
        length = options.maxIter
    }
    let {Ynorm, R, nu, nm, nf, lambda} = arguments
    const RHO = 0.01
    const SIG = 0.5
    const INT = 0.1
    const EXT = 3.0
    const MAX = 20
    const RATIO = 10
    

    let red = 1
    
    let i = 0
    let ls_failed = 0
    let fX = []
    let A = 0
    let B = 0
    let z2 = 0
    //let fX = tf.tensor([])
    //fX.print()
    //console.log('Primer f1')
    let arr = await f(X, Ynorm, R, nu, nm, nf, lambda)
    let f1 = arr[0]
    let df1 = arr[1]
    //console.log('f1', f1.arraySync())
    i = i + (length<0)
    //console.log('df1')
    //df1.print()//350x1
    let s = tf.mul(df1, -1)
    //console.log('s')
    //s.print()//350x1
    let d1 = tf.matMul(tf.mul(s, -1), s, true, false).reshape([])
    //console.log('d1', d1.arraySync())   
    let z1 = tf.div(red, tf.sub(1, d1))
    //console.log('z1', z1.arraySync())
    

    while(i < Math.abs(length)){
        i = i + (length>0)
        //console.log('i', i)
        let X0 = X
        let f0 = f1
        let df0 = df1

        X = tf.add(X, tf.mul(z1, s))
        //X.print()
        //console.log('Segun f2')
        arr = await f(X, Ynorm, R, nu, nm, nf, lambda)
        let f2 = arr[0]
        let df2 = arr[1]
        //console.log('f2', f2.arraySync())
        //console.log('df2')
        //df2.print()
        i = i + (length<0)
        //console.log(i)
        let d2 = tf.matMul(df2, s, true, false).reshape([])
        //console.log('d2', d2.arraySync())
        let f3 = f1
        let d3 = d1
        let z3 = tf.mul(-1, z1)
        //console.log('f3', f3.arraySync())
        //console.log('d3', d3.arraySync())
        //console.log('z3', z3.arraySync())

        if(length > 0){
            var M = MAX
        }else{
            var M = Math.min(MAX, -length-i)
        }
        //console.log('M ', M)
        let success = 0
        let limit = -1
        //console.log('limit', limit)
        while(1){
            //console.log('firstWhile', (tf.greater(f2, tf.add(f1, tf.mul(z1, tf.mul(RHO, d1)))).arraySync() || tf.greater(d2, tf.mul(-SIG, d1)).arraySync()) && M > 0)
            //console.log('conditionOne', tf.greater(f2, tf.add(f1, tf.mul(z1, tf.mul(RHO, d1)))).arraySync())
            //console.log('conditionTwo', tf.greater(d2, tf.mul(-SIG, d1)).arraySync())
            while((tf.greater(f2, tf.add(f1, tf.mul(z1, tf.mul(RHO, d1)))).arraySync() || tf.greater(d2, tf.mul(-SIG, d1)).arraySync()) && M > 0){
                limit = z1 
                //console.log('limit', limit.arraySync())   
                if(tf.greater(f2, f1).arraySync()){
                    //console.log('Entra z1')
                    z2 = tf.sub(z3, tf.div(tf.mul(0.5, tf.mul(d3, tf.mul(z3, z3))), tf.add(tf.mul(d3, z3), tf.sub(f2, f3))))
                    //console.log('z2', z2.arraySync())
                }else{
                    //console.log('Entra z2')
                    A = tf.add(tf.mul(6, tf.div(tf.sub(f2, f3), z3)), tf.mul(3, tf.add(d2, d3))) 
                    B = tf.sub(tf.mul(3, tf.sub(f3, f2)), tf.mul(z3, tf.add(d3, tf.mul(2, d2))))
                    z2 = tf.div(tf.sub(tf.sqrt(tf.sub(tf.mul(B, B), tf.mul(A, tf.mul(d2, tf.mul(z3, z3))))), B), A)     
                    //console.log('z2', z2.arraySync())
                }

                if(tf.isNaN(z2).arraySync() || tf.isInf(z2).arraySync()){
                    //console.log('Entra NaN')
                    z2 = tf.div(z3, 2)
                    //console.log('z2', z2.arraySync())
                    //z1.print()
                }

                //tf.minimum(z2, tf.mul(INT, z3)).print()
                //tf.mul(tf.sub(1,INT), z3).print()
                z2 = tf.maximum(tf.minimum(z2, tf.mul(INT, z3)), tf.mul(tf.sub(1,INT), z3))
                //z2.print()
                z1 = tf.add(z1, z2)
                //z1.print()
                X = tf.add(X, tf.mul(z2, s))
                arr = await f(X, Ynorm, R, nu, nm, nf, lambda)
                // console.log('Tercer')
                f2 = arr[0]
                df2 = arr[1]
                M = M - 1
                i = i + (length<0)
                d2 = tf.matMul(df2, s, true, false).reshape([])
                z3 = tf.sub(z3, z2)
                //z3.print()
                //cond1 = tf.less(f2, tf.add(f1, tf.mul(z1, tf.mul(RHO, d1)))).arraySync()[0][0]
                //cond2 = tf.greater(d2, tf.mul(-SIG, d1)).arraySync()[0][0] && M > 0
            }
            //console.log('Sale while')
            if((tf.greater(f2, tf.add(f1, tf.mul(z1, tf.mul(RHO, d1)))).arraySync() || tf.greater(d2, tf.mul(-SIG, d1)).arraySync())){
                //console.log('Un break')
                break
            }else if(tf.greater(d2, tf.mul(SIG, d1)).arraySync()){
                //console.log('Dos break')
                success = 1
                break
            }else if(M === 0){
                //console.log('Tres Break')
                break
            }  
            //console.log(tf.mul(6, tf.div(tf.sub(f2, f3),z3)).arraySync())
            //console.log(tf.mul(3, tf.add(d2, d3)).arraySync())
            A = tf.add(tf.mul(6, tf.div(tf.sub(f2, f3), z3)), tf.mul(3, tf.add(d2, d3))) 
            //console.log('A', A.arraySync())
            //console.log(tf.mul(3, tf.sub(f3, f2)).arraySync())
            //console.log(tf.mul(z3, tf.add(d3, tf.mul(2, d2))).arraySync())
            B = tf.sub(tf.mul(3, tf.sub(f3, f2)), tf.mul(z3, tf.add(d3, tf.mul(2, d2))))
            //console.log('B', B.arraySync())
            //console.log(tf.mul(-1, tf.mul(d2, tf.mul(z3, z3))).arraySync())
            //console.log(tf.add(B, tf.sqrt(tf.sub(tf.mul(B, B), tf.mul(A, tf.mul(d2, tf.mul(z3, z3)))))).arraySync())
            z2 = tf.div(tf.mul(-1, tf.mul(d2, tf.mul(z3, z3))), tf.add(B, tf.sqrt(tf.sub(tf.mul(B, B), tf.mul(A, tf.mul(d2, tf.mul(z3, z3)))))))
            //console.log('z2', z2.arraySync()) 
            //console.log(limit)
            //console.log(limit > -0.5)
            //console.log(tf.isNaN(z2).arraySync() || tf.isInf(z2).arraySync() || tf.less(z2, 0).arraySync())      
            //console.log(tf.greater(tf.add(z2, z1), limit).arraySync())  
            //console.log(tf.greater(tf.add(z2, z1), tf.mul(z1, EXT)).arraySync())  
            //console.log(tf.less(z2, tf.mul(-1, tf.mul(z3, INT))).arraySync()) 
            //console.log(tf.less(z2, tf.mul(tf.sub(limit, z1), tf.sub(1.0, INT))).arraySync())     
            if(tf.isNaN(z2).arraySync() || tf.isInf(z2).arraySync() || tf.less(z2, 0).arraySync()){
                if(limit < -0.5){
                    z2 = tf.mul(z1, tf.sub(EXT, 1))
                    //console.log('z2', z2.arraySync())
                }else{
                    z2 = tf.div(tf.sub(limit, z1), 2)
                    //console.log('z2', z2.arraySync())
                }
            }else if(limit > -0.5 && tf.greater(tf.add(z2, z1), limit).arraySync()) {
                //console.log('Entra 2')
                z2 = tf.div(tf.sub(limit, z1), 2)
                //console.log('z2', z2.arraySync())
            }else if(limit < -0.5 && tf.greater(tf.add(z2, z1), tf.mul(z1, EXT)).arraySync()){
                //console.log('Entra 3')
                z2 = tf.mul(z1, tf.sub(EXT, 1.0))
                //console.log('z2', z2.arraySync())
            }else if(tf.less(z2, tf.mul(-1, tf.mul(z3, INT))).arraySync()){
                //console.log('Entra 4')
                z2 = tf.mul(-1, tf.mul(z3, INT))
                //console.log('z2', z2.arraySync())
            }else if(limit > -0.5 && tf.less(z2, tf.mul(tf.sub(limit, z1), tf.sub(1.0, INT))).arraySync()) {                        
                //console.log('Entra 5')
                z2 = tf.mul(tf.sub(limit, z1), tf.sub(1.0, INT))
                //console.log('z2', z2.arraySync())
            }
            f3 = f2
            //console.log('f3', f3.arraySync())
            d3 = d2
            //console.log('d3', d3.arraySync())
            z3 = tf.mul(-1, z2)
            //console.log('z3', z3.arraySync())
            z1 = tf.add(z1, z2)
            //console.log('z1', z1.arraySync())
            X = tf.add(X, tf.mul(z2, s))
            arr = await f(X, Ynorm, R, nu, nm, nf, lambda)
            //console.log('Cuart')
            f2 = arr[0]
            //console.log('f2', f2.arraySync())
            df2 = arr[1]
            M = M-1
            //console.log('M', M)
            i = i + (length < 0)
            //console.log('i', i)
            d2 = tf.matMul(df2, s, true, false).reshape([])
            //console.log('d2', d2.arraySync())
        }
        //console.log('success', success)
        if(success){
            f1 = f2
            fX.push(f1.arraySync())
            console.log(`Cost ${i}: ${f1.arraySync()}`)
            //console.log(fX)
            s = tf.sub(tf.mul(tf.div(tf.sub(tf.matMul(df2, df2, true, false), tf.matMul(df1, df2, true, false)), tf.matMul(df1, df1, true, false)), s), df2)
            //s.print()
            let tmp = df1
            df1 = df2
            df2 = tmp
            d2 = tf.matMul(df1, s, true, false).reshape([])
            //console.log(tf.greater(d2, 0).arraySync())
            if(tf.greater(d2, 0).arraySync()){
                s = tf.mul(-1, df1)
                d2 = tf.matMul(tf.mul(-1, s), s, true, false).arraySync()
                //console.log('d2', d2.arraySync())
            }
            z1 = tf.mul(z1, tf.minimum(RATIO, tf.div(d1, tf.sub(d2, -99999999999999))))
            //console.log('z1', z1.arraySync())
            d1 = d2
            //console.log('d1', d1.arraySync())
            ls_failed = 0
        }else{
            X = X0
            f1 = f0
            df1 = df0
            if(ls_failed || i > Math.abs(length)){
                break
            }
            let tmp = df1
            df1 = df2
            df2 = tmp
            s = tf.mul(-1, df1)
            d1 = tf.matMul(tf.mul(-1,s), s, true, false).reshape([])
            //console.log('d1', d1.arraySync())
            z1 = tf.div(1, tf.sub(1, d1))
            //console.log('z1', z1.arraySync())
            ls_failed = 1
        }
    
    }
    return [X, fX, i]
}

module.exports = fmincg