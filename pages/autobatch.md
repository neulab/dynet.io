---
title: Autobatch
layout: page
permalink: /autobatch/
---

## Friends don't let friends write batching code

Modern hardware processors (CPUs and GPUs) can use parallelism to a great extent. So batching is good for speed. But it is so annoying to write batching code for RNNs or more complex architectures. You must take care of padding, and masking, and indexing, and that's just for the easy cases... Not any more!

We've added a feature to DyNet that will transform the way you think about and run batching code. The gist of it is: you aggregate a large enough computation graph to make batching possible. DyNet figures out the rest, and does the batching for you.

{:style="text-align: center;"}
![An Example of Autobatching](https://github.com/clab/dynet/raw/37f61e6c21dad11297d30a34066931a0a792ca0c/examples/python/tutorials/imgs/autobatch.gif){:style="width: 75%;"}

In what follows, we show some examples of non-batched DyNet code, and then move on to show the batched version.

In order to enable auto-batching support, simply add --dynet-autobatch 1 to the commandline flags when running a DyNet program. Check out the paper or read on for more details!

## Dynamic Graphs, Non-batched
Let's look at some examples of non-batched code, and how simple they are to write in DyNet.

Our first example will be an acceptor LSTM, that reads in a sequence of vectors, passes the final vector through a linear layer followed by a softmax, and produces an output.

{% highlight python %}
    import dynet as dy
    import numpy as np

    # acceptor LSTM
    class LstmAcceptor(object):
        def __init__(self, in_dim, lstm_dim, out_dim, model):
            self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
            self.W       = model.add_parameters((out_dim, lstm_dim))

        def __call__(self, sequence):
            lstm = self.builder.initial_state()
            W = self.W.expr() # convert the parameter into an Expession (add it to graph)
            outputs = lstm.transduce(sequence)
            result = W*outputs[-1]
            return result

    # usage:
    VOCAB_SIZE = 1000
    EMBED_SIZE = 100

    m = dy.Model()
    trainer = dy.AdamTrainer(m)
    embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
    acceptor = LstmAcceptor(EMBED_SIZE, 100, 3, m)


    # training code
    sum_of_losses = 0.0
    for epoch in range(10):
        for sequence,label in [((1,4,5,1),1), ((42,1),2), ((56,2,17),1)]:
            dy.renew_cg() # new computation graph
            vecs = [embeds[i] for i in sequence]
            preds = acceptor(vecs)
            loss = dy.pickneglogsoftmax(preds, label)
            sum_of_losses += loss.npvalue()
            loss.backward()
            trainer.update()
        print sum_of_losses / 3
        sum_of_losses = 0.0

    print "\n\nPrediction time!\n"
    # prediction code:
    for sequence in [(1,4,12,1), (42,2), (56,2,17)]:
        dy.renew_cg() # new computation graph
        vecs = [embeds[i] for i in sequence]
        preds = dy.softmax(acceptor(vecs))
        vals  = preds.npvalue()
        print np.argmax(vals), vals
{% endhighlight %}

{% highlight python %}
    [ 1.1042192]
    [ 1.03213656]
    [ 0.97442627]
    [ 0.91803074]
    [ 0.86030102]
    [ 0.79953943]
    [ 0.73457642]
    [ 0.66490026]
    [ 0.59101043]
    [ 0.51482052]
{% endhighlight %}


Prediction time!

{% highlight python %}
    1 [ 0.06114297  0.75843614  0.18042086]
    1 [ 0.25732863  0.37167525  0.37099609]
    1 [ 0.1679846   0.61701268  0.21500272]
{% endhighlight %}

This was simple. Notice how each sequence has a different length, but its OK, the LstmAcceptor doesn't care. We create a new graph for each example, at exactly the desired length.

Similar to the LstmAcceptor, we could also write a TreeRNN that gets as input a tree structure and encodes it as a vector. Note that the code below is missing the support code for rerpesenting binary trees and reading trees from bracketed notation. All of these, along with the more sophisticated TreeLSTM version, and the training code, can be found here.

{% highlight python %}
    class TreeRNN(object):
        def __init__(self, model, word_vocab, hdim):
            self.W = model.add_parameters((hdim, 2*hdim))
            self.E = model.add_lookup_parameters((len(word_vocab),hdim))
            self.w2i = word_vocab

        def __call__(self, tree): return self.expr_for_tree(tree)

        def expr_for_tree(self, tree):
            if tree.isleaf():
                return self.E[self.w2i.get(tree.label,0)]
            if len(tree.children) == 1:
                assert(tree.children[0].isleaf())
                expr = self.expr_for_tree(tree.children[0])
                return expr
            assert(len(tree.children) == 2),tree.children[0]
            e1 = self.expr_for_tree(tree.children[0], decorate)
            e2 = self.expr_for_tree(tree.children[1], decorate)
            W = dy.parameter(self.W)
            expr = dy.tanh(W*dy.concatenate([e1,e2]))
            return expr
{% endhighlight %}


## Enter batching

Now, let's add some minibatching support. The way we go about it is very simple: Your only responsibility, as a programmer, is to *build a computation graph with enough material to make batching possible* (i.e., so there is something to batch). DyNet will take care of the rest.

Here is the training and prediction code from before, this time writen with batching support. Notice how the LstmAcceptor did not change, we just aggregate the loss around it.

{% highlight python %}
    # training code: batched.
    for epoch in range(10):
        dy.renew_cg()     # we create a new computation graph for the epoch, not each item.
        # we will treat all these 3 datapoints as a single batch
        losses = []
        for sequence,label in [((1,4,5,1),1), ((42,1),2), ((56,2,17),1)]:
            vecs = [embeds[i] for i in sequence]
            preds = acceptor(vecs)
            loss = dy.pickneglogsoftmax(preds, label)
            losses.append(loss)
        # we accumulated the losses from all the batch.
        # Now we sum them, and do forward-backward as usual.
        # Things will run with efficient batch operations.
        batch_loss = dy.esum(losses)/3
        print batch_loss.npvalue() # this calls forward on the batch
        batch_loss.backward()
        trainer.update()

    print "\n\nPrediction time!\n"
    # prediction code:
    dy.renew_cg() # new computation graph
    batch_preds = []
    for sequence in [(1,4,12,1), (42,2), (56,2,17)]:
        vecs = [embeds[i] for i in sequence]
        preds = dy.softmax(acceptor(vecs))
        batch_preds.append(preds)

    # now that we accumulated the prediction expressions,
    # we run forward on all of them:
    dy.forward(batch_preds)
    # and now we can efficiently access the individual values:
    for preds in batch_preds:
        vals  = preds.npvalue()
        print np.argmax(vals), vals
{% endhighlight %}

{% highlight python %}
    [ 0.46247479]
    [ 0.43548316]
    [ 0.40905878]
    [ 0.38335174]
    [ 0.35849127]
    [ 0.3345806]
    [ 0.31169581]
    [ 0.28988609]
    [ 0.26917794]
    [ 0.24957809]
{% endhighlight %}


Prediction time!

{% highlight python %}
    1 [ 0.00736407  0.95775431  0.03488157]
    2 [ 0.2252606   0.36341026  0.41132909]
    1 [ 0.05491769  0.85925961  0.08582276]
{% endhighlight %}

Doing the same thing for the TreeRNN example is trivial: just aggregate the expressions from several trees, and then call forward. (In fact, you may receive a small boost from the auto-batching feature also within a single tree, as some computation can be batched there also.)

## Comparison to manual batching

We compared the speed of automatic-batching as shown above to a manualy crafted batching code, in a setting in which manual-batching excels: BiLSTM tagging where all the sentences are of the exact same length. Here, automatic batching improved the per-sentence computation time from 193ms to 16.9ms on CPU and 54.6ms to 5.03ms on GPU, resulting in an approximately 11-fold increase in sentences processed per second (5.17->59.3 on CPU and 18.3->198 on GPU). However, manual batching is still 1.27 times faster on CPU, and 1.76 times faster on a GPU.

The speed in favor of manual batching seem to come mostly from the time it takes to create the computation graph itself: in manual batching we are creating a single graph with many inputs, while with automatic batching we essentially build many copies of the same graph for each batch. Should you use manual batching then? In situations in which it is very natural, like in this artificial one, sure! But in cases where manual batching is not so trivial (which is most cases, see some examples below), go ahead and use the automatic version. It works.

You can also run automatic batching on top of manually batched code. When doing this, we observe another 10% speed increase above the manual batched code, when running on the GPU. This is because the autobatching engine managed to find and exploit some additional batching opportunities. On the CPU, we did not observe any gains in this setting, but also did not observe any losses.

## How big is the win?

So the examples above are rather simple, but how does this help on actual applications? We've run some experiments on several natural language processing tasks including POS tagging with bidirectional LSTMs, POS tagging with BiLSTMs that also have character embeddings (which is harder to batch), tree-structured neural networks, and a full-scale transition-based dependency parser. Each of these has a batch size of 64 sentences at a time, without worrying about length balancing or anything of that sort. As you can see from the results below on sentences/second, auto-batching gives you healthy gains of 3x to 9x over no auto-batching. This is with basically no effort required!

{:class="table table-bordered table-sm bg-light"}
|:------------------|:-------------------|:----------------|:-------------------|:----------------|
| Task              | No Autobatch (CPU) | Autobatch (CPU) | No Autobatch (GPU) | Autobatch (GPU) |
|:------------------|:-------------------|:----------------|:-------------------|:----------------|
| BiLSTM            | 16.8               | 156             | 56.2               | 367             |
| BiLSTM w/ char    | 15.7               | 132             | 43.2               | 275             |
| TreeNN            | 50.2               | 357             | 76.5               | 661             |
| Transition Parser | 16.8               | 61.2            | 33.0               | 90.1            |
|-------------------|--------------------|-----------------|--------------------|-----------------|

If you want to try these benchmarks yourself, take a look at the ...-bulk programs in the dynet-benchmark repository.
In the graph below you can see the number of sentences/second for training the transition-based parser with various batch sizes, on the GPU, CPU, and CPU witk MKL enabled:

{:style="text-align: center;"}
![Autobatching Speed in Various Batch Sizes](https://github.com/clab/dynet/raw/37f61e6c21dad11297d30a34066931a0a792ca0c/examples/python/tutorials/imgs/bist-autobatch-speed.png){:style="width: 50%;"}

The following graph shows the number of sentences/second for the Tree-LSTM model for various batch sizes, and also compares to TensorFlow Fold implementation, which is another proposed solution for batching hard-to-batch architectures. Note that DyNet autobatching comfortably wins over TensorFlow fold for both GPU and CPU, with CPU being more efficient than GPU for smaller sized batches.

{:style="text-align: center;"}
![Autobatching Speed in Various Batch Sizes](https://github.com/clab/dynet/raw/37f61e6c21dad11297d30a34066931a0a792ca0c/examples/python/tutorials/imgs/treelstm-autobatch-speed.png){:style="width: 50%;"}

## Miscellaneous tips

### Should you always use batching?

It depends. In prediction time, batching is a pure win in terms of speed. In training time, the sentences/second throughput will be much better---but you will also have less parameter updates, which may make overall training slower. Experiment with different batch sizes to find a good tradeoff between the two.

### Length-balanced batches?

It is common knowledge when writing batched code that one should arrange the batches such that all examples within the batch are of the same size. This is crucial for static frameworks and manual batching, as it reduces the need for padding, masking and so on. In our framework, this is not needed. However, you may still win some speed by having relatively-balanced batches, because more batching opportunities will become available.

### Tips for effective autobatching

As we said above, our only rule is "create a graph with enough material for the autobatcher to work with". In other words, it means delaying the call to forward() (or to value(), npvalue(), scalar_value()...) as much as possible. Beyond that, things should be transparent.

However, knowing some technicalities of DyNet and how forward works can help you avoid some pitfals. So here is a brief overview:

  1. The core building block of dynet are Expression objects. Whenever you create a new Expression, you extend the computation graph.
  2. Creating an Expression does not entail a forward computation. We only evaluate the graph when specifically asked for it.
  3. Calls to e.forward(), e.value(), e.npvalue(), e.scalar_value(), will run forward computation up to that expression, and return a value.
  4. These calls will compute all the expressions that were added to the graph before e. These intermediary results will be cached.
  5. Asking for values for (or calling forward on) earlier expressions, will reuse the cached values.
  6. You can extend the graph further after calling forward. Later calls will compute the graph delta.

So, based on this knowledge, here is the rule:

If you created several expressions, and want to get the values for them, call forward on the last expression first, and then on the previous ones.

Doing it the other way around (evaluting the expressions in the order they were created) will hinder batching possibilities, because it will compute only a small incremental part of forward for each expression. On the other hand, if you run forward on the last expression first, the entire computation will happen in one chunk, batching when possible. Getting calling npvalue() on the earlier expressions will then return the already computed values.

If you created a bunch of expressions and are not sure which one is the latest, you could just call the special list version of forward:

{% highlight python %}
    dy.forward([e1,e2,...,en])
{% endhighlight %}

and it will figure it out for you.

## Loose ends

Auto-batching in DyNet works and is stable. However, some of the less common operations are not yet batched. If you have an example where you think you should be getting a nice boost from autobatching but you don't, it is most likely that you are using a non-batched operation. In any case, let us know via an issue in github, and we'll investigate this.
