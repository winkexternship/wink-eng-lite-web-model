var summary = function ( rdd, its, as, simmilarity, bm25 ) {

    var tokens = rdd.tokens;
    var sentences = rdd.sentences;
    var cache = rdd.cache;
    var numOfSentences = rdd.sentences.length;
    var aptTokens = [];
    var bow = [];
    var aptPOS = [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];
    var convergenceThreshold =  0.0001;
    var maxIteration = 100;
    var dampingFactor = 0.85;
    var numOfConvergence = 0;
    var colSum = []; // Normalisation Parameter Method 2
    var nonZero = []; // Normalisation Parameter Method 1
    var ranks = []; // Used for pagerank without node weights
    var weights = []; // Used for pagerank with node weights

    console.log(`Number of sentences in text:${numOfSentences}`);
    // Getting appropriate tokens for text summarization (Preprocessing)
    for ( let i = 0; i < numOfSentences; i += 1) {
        const temp = [];
        for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1) {
            if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
                temp.push(its.normal( j, tokens, cache ));
            }
        }
        // bow will be used if we are using cosine simmilarity funtion
        // bow.push(as.bow(temp));

        // used for common terms
        aptTokens.push(temp);
    }

    // bm25 learning and creating bow
    aptTokens.forEach( colToken => bm25.learn(colToken) );
    for ( let i=0; i<aptTokens.length; i+=1 ) {
        bow.push(bm25.doc(i).out(its.bow));
    }
    console.log(bow);

    console.log(`Number of sentences in aptTokens:${aptTokens.length}`);
    console.log('aptTokens:');
    console.log(aptTokens);

    // Creating adjacency matrix for graph of sentences
    var senGraph = new Array(numOfSentences);
    for ( let i = 0; i < numOfSentences; i += 1) {
        senGraph[i] = new Array(numOfSentences);
    }

    // Populating the matrix with weights
    for ( let i = 0; i < numOfSentences; i += 1) {
        senGraph[i][i] = 0;
        for ( let j = i + 1; j < numOfSentences; j += 1) {
            // change the following to implement different ways to calculate simmilarity value
            const numOfCommonTokens = aptTokens[i].filter((token) => aptTokens[j].includes(token)).length;
            senGraph[i][j] = numOfCommonTokens / (Math.log(aptTokens[i].length) + Math.log(aptTokens[j].length));
            // For cosine simmilarity
            // senGraph[i][j] = simmilarity.bow.cosine(bow[i], bow[j]);
            if( !Number.isFinite (senGraph[i][j] )  )
                senGraph[i][j] = 0;
            senGraph[j][i] = senGraph[i][j];
        }
    }
    console.log('SenGraph:');
    console.log(senGraph);

    // Page Rank (Calculating Normalization Parameter Method 1) 
    for ( let i = 0; i < numOfSentences; i += 1) {
        nonZero[i] = senGraph[i].reduce((previousValue) =>  previousValue + 1 ,0);
    }

    // Page Rank (Calculating Normalization Parameter Method 2)
    // for ( let i = 0; i < numOfSentences; i += 1) {
    //     colSum[i] = senGraph[i].reduce((previousValue, currentValue) =>  previousValue + currentValue ,0);
    // }

    // Page Rank (Method 1 without Node Weights)
    for ( let i = 0; i < numOfSentences; i += 1 ) {
        const s = senGraph[i].reduce((previousValue, currentValue, idx) =>  previousValue + ( currentValue / nonZero[idx] )  ,0);
        ranks.push({ idx: i, val: s });
    }

    // Page Rank (Method 2 with Node Weights)
    // Initializing the weights vector (For PageRank Method 2)
    // for ( let i = 0; i < numOfSentences; i += 1) {
    //     weights.push({ idx: i, val: 1 / numOfSentences });
    // }
    // console.log('Intital Weights:');
    // console.log(weights);
    // for ( let i = 0; i < maxIteration; i += 1) {
    //     for ( let j = 0; j < numOfSentences; j += 1) {
    //         let rank = 1 - dampingFactor;
    //         rank += dampingFactor * senGraph[j].reduce((previousValue, currentValue, idx) => previousValue + ((currentValue / colSum[idx]) * weights[idx].val) ,0);
    //         if ( Math.abs(weights[j] - rank) <= convergenceThreshold ) {
    //             numOfConvergence += 1;
    //         }
    //         weights[j].val = rank;
    //     }
    //     if ( numOfConvergence === numOfSentences ) {
    //         break;
    //     }
    // }
    
    // Sentence Selection
    ranks.sort( (a, b) => b.val - a.val ); //For PageRank Method 1
    // weights.sort( (a, b) => b.val - a.val ) //For Pagerank Method 2
    console.log('Final Weights:');
    console.log(ranks);
    
    return {
        weights: ranks,
        numOfSentences: numOfSentences
    };

    // return 'This is a summary text for the given input text.';
};

module.exports = summary;
