<TeXmacs|1.99.2>

<style|generic>

<\body>
  <section|Overview>

  <strong|Idea.> Trying to get the maximum from datsets where we have very
  few training examples, but each example has a very large number of
  features. Examples of such datasets include medical databases where we have
  gene activation measurements for very few patients but many different
  genes.

  <strong|Method.> We design a neural network architecture whose number of
  parameters is constant with respect to the number of features (which is not
  the case with a typical linear classifier). The basic idea is that we use a
  linear classifier whose coefficients for each features are generated by a
  single neural nework that takes as an input a representation for this
  feature, which is basically a transformation of the set of values taken by
  this feature through ale the exampes. More complex (deep) architectures are
  also experimented.

  <strong|Datasets.>

  <\itemize>
    <item>ICML 2003 feature selection challenge datasets: Arcene, Dorothea

    <item>AML/ALL Leukemia classification dataset
  </itemize>

  <section|Models & implementation>

  <strong|Notations.> We call <math|X> the matrix of training examples, where
  each line of <math|X> is an example, and each column <math|r<rsub|j>> of
  <math|X> is the column of all the values taken by a feature throughout the
  training set. <math|n> is the number of lines of <math|X> (number of
  training examples) and <math|d> is the number of columns of <math|X>
  (number of features). We call <math|Y> the column matrix of training labels
  <math|y<rsub|i>>.

  <strong|Code.> The code uses Blocks and Fuel frameworks developped at MILA.
  The <verbatim|dataset.py> and <verbatim|datastream.py> files are
  responsible for loading the datasets into Fuel. The <verbatim|train.py>
  script is the main script of the project, that loads the data and the model
  and does the training. <verbatim|train.py> expects one argument, which is
  the name of a model to do the training on. For instances, existing models
  include <verbatim|mlp.py>, <verbatim|mlpfsel.py>, <verbatim|mlpfsel2.py>,
  etc. To run the model from <verbatim|mlp.py>, type:

  <\verbatim-code>
    $ ./train.py mlp
  </verbatim-code>

  <subsection|First idea>

  This is a first model for binary classification. It can probably be
  extended to <math|n>-ary classification.

  <strong|Model.> The example <math|x<rsub|>> is decomposed in its features
  <math|x<rsub|j>>. We have a function <math|f<around*|(|x<rsub|j>,r<rsub|j>|)>>
  which is a MLP that calculates a probability that the example is a positive
  sample. We then have them vote to produce classification <math|<wide|y|^>>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|y<rsub|>|^>>|<cell|=>|<cell|<big|sum><rsub|j=1><rsup|d>f<around*|(|x<rsub|j>,r<rsub|j>|)>>>>>
  </eqnarray*>

  During training, <math|x> is one of the rows of <math|X>. During validation
  and testing, different examples are chosen. The parameters of the model are
  the parameters of the MLP implemented by <math|f>.

  <strong|Implementation.> A simple implementation of this model is in the
  repo, in <verbatim|mlp.py>.

  <subsection|Recurrent neural nets>

  <strong|Model.> We do the same as previously, but only the NN is a
  recurrent net that scans the features successively in a random order. A
  prediction is output at each timestep.

  <strong|Implementation.> A simple implementation is available in
  <verbatim|rnn.py> that uses LSTM. A more complex implementation in
  <verbatim|mlprnn.py> first passes <math|<around*|(|x<rsub|j>,r<rsub|j>|)>>
  through a MLP, which can be usefull for dimension reduction.

  <subsection|MLP-generated linear classifier>

  This model is also described for binary classification and can also be
  extended to <math|n>-ary classification.

  <strong|Model.> We have a function <math|f<around*|(|r<rsub|j>|)>> which
  for each features gives a single coefficient in a linear classifier:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|y<rsub|>|^>>|<cell|=>|<cell|\<sigma\><around*|(|<big|sum><rsub|j=1><rsup|d>f<around*|(|r<rsub|j>|)>*x<rsub|j>+b|)>>>>>
  </eqnarray*>

  where <math|\<sigma\>> is a sigmoid function. The parameters of the model
  are the bias <math|b> and the parameters defining the MLP <math|f>.

  <strong|Implementation.> A simple implementation of this model is in
  <verbatim|mlpfsel.py>.

  <subsection|MLP-generated MLP>

  This is the model that has been experimented with the most.

  <strong|Model.> The idea is that the function
  <math|f<around*|(|r<rsub|j>|)>> gives us not only one coefficient for each
  feature, but a bunch of them, producing the matrix <math|W<rsub|0>> of
  weights for the first layer of a MLP:

  <\eqnarray*>
    <tformat|<table|<row|<cell|W<rsub|0>>|<cell|=>|<cell|<around*|(|<stack|<tformat|<table|<row|<cell|f<around*|(|r<rsub|1>|)>>>|<row|<cell|*\<vdots\>>>|<row|<cell|f<around*|(|r<rsub|d>|)>>>>>>|)>>>|<row|<cell|<wide|y|^>>|<cell|=>|<cell|g<around*|(|x*W<rsub|0><rsup|\<top\>>|)>>>>>
  </eqnarray*>

  Both <math|f> and <math|g> are MLPs whose coefficients are the parameters
  of the model.

  <strong|Implementation.> Available in <verbatim|mlpfsel2.py>

  <strong|Extensions.>

  <\itemize>
    <item><verbatim|mlpfsel3.py> : <math|f> is decomposed in two successive
    MLPs, and a reconstruction penalty is applied so that from the output of
    the first half of <math|f<around*|(|r<rsub|j>|)>> we can reconstruct the
    full vector <math|r<rsub|j>> (through yet another MLP).

    <item><verbatim|mlpfsel4.py> : we divide the features <math|r<rsub|j>>
    into several (possibly overlapping) subsets, and then have a separate
    classificaion system <math|f<rsup|<around*|(|k|)>>,g<rsup|<around*|(|k|)>>>
    for each subset:

    <\eqnarray*>
      <tformat|<table|<row|<cell|r<rsub|j><rsup|<around*|(|k|)>>>|<cell|=>|<cell|\<pi\><rsup|<around*|(|k|)>><around*|(|r<rsub|j>|)>>>|<row|<cell|W<rsub|0><rsup|<around*|(|k|)>>>|<cell|=>|<cell|<around*|(|<stack|<tformat|<table|<row|<cell|f<rsup|<around*|(|k|)>><around*|(|r<rsub|1><rsup|<around*|(|k|)>>|)>>>|<row|<cell|\<vdots\>>>|<row|<cell|f<rsup|<around*|(|k|)>><around*|(|r<rsub|d><rsup|<around*|(|k|)>>|)>>>>>>|)>>>|<row|<cell|<wide|y|^><rsup|<around*|(|k|)>>>|<cell|=>|<cell|g<rsup|<around*|(|k|)>><around*|(|x*W<rsub|0><rsup|<around*|(|k|)>\<top\>>|)>>>|<row|<cell|<wide|y|^>>|<cell|=>|<cell|<frac|1|K>*<big|sum><rsub|k=1><rsup|K><wide|y|^><rsup|<around*|(|k|)>>>>>>
    </eqnarray*>

    <item><verbatim|mlpfsel5.py> : combination of the approaches from
    <verbatim|mlpfsel3> and <verbatim|mlpfsel4>.

    <item><verbatim|mlpfsel2ae.py> : a deep auto-encoder is first pre-trained
    to encode the <math|r<rsub|j>> vectors into a more compact/sparse
    representation, that is then fed to <math|f>.

    <item>In <verbatim|mlpfsel2.py>, principal component analysis can be done
    on the <math|r<rsub|j>> vectors so that <math|r<rsub|j>> are replaced by
    their projectons on the <math|m> most present components.

    <item>Various forms of regularization (dropout, noise, L1 penalty) are
    sometimes implemented in all the models.
  </itemize>

  <section|Current results>

  <strong|Convergence.> It is hard to make any of these models converge. The
  only optimization algorithm that we have managed to use is AdaDelta. A
  simple SGD momentum does work for some models (<verbatim|mlp>,
  <verbatim|mlprnn>) but it stays on a plateau for a very long time before
  the costs goes down. RMSProp does not work at all: in all our experiments
  the cost diverges and the model parameters become meaningless.

  <strong|Baseline.> For the NIPS 2003 feature selection challenge, previous
  results are available at <hlink|this URL|http://web.archive.org/web/20130512034606/http://www.nipsfsc.ecs.soton.ac.uk/datasets>.
  An online judge for trying solutions against the undisclosed test set is
  available <hlink|here|https://www.codalab.org/competitions/3931?secret_key=d6c218a3-3b83-4eed-8e39-5b895c5a5e35#learn_the_details>.

  <strong|Overfitting.> A typical behaviour for all the models is
  overfitting. The training costs and error rate typically go to zero, while
  the validation cost and error rate diverge. For instance this is what we
  observe on Arcene, using 100 training examples and 100 validation examples:

  <\itemize>
    <item>The validation cost starts at 0.6 or 0.7, goes down a bit, then
    back up. When the model is settled at zero error for the training, the
    validation cost continues a regular upward progression going often way
    above 1.

    <item>The validation error rate goes down a bit and stabilizes above 15%,
    often above 20%. Best hyperparameter choices have converged at 13%
    validation error. Submissions to the online judge have yielded balanced
    error rates above 15% for the undisclosed test set, which is far from the
    best entries that manage to do 7% error rate.
  </itemize>

  <strong|Regularization.>

  <\itemize>
    <item>Noise on weights and on <math|r<rsub|j>> inputs can be used but
    does not bring us to exceptionnal performance. Too much noise makes the
    model fluctuate and the validation costs do not converge anymore.

    <item>Dropout does not help us much either, and too much dropout has the
    same effect as too much noise.

    <item>In the <verbatim|mlpfsel2> model, applying a L1 penalty on the
    weights of the first MLP (<math|f>) is effective to prevent the
    validation cost from diverging, but does not improve the error rates.
  </itemize>

  <strong|Hyperparameter search.> Given the number of different models and
  the number of hyperparameters that can be tweaked, the hyperparameter
  search has been extremely cumbersome and no definitive sweeet spot has yet
  been identified.

  <strong|Best solution at the moment.> The best solution found at the moment
  that brings the validation cost to 13% on Arcene is based on the following
  model:

  <\itemize>
    <item><strong|<verbatim|mlpfsel2>> model, no hidden layer in <math|f>,
    <math|f> outputs 10 coefficients, no hidden layer in <math|g>

    <item>pre-processing is a PCA on the vectors <math|r<rsub|j>> and about
    50 out of 100 components are kept

    <item>a L1 penalty of 0.01 is applied on the coefficients of <math|f>,
    and a L1 penalty of 0.001 is applied on the input to <math|g>
  </itemize>

  This solution has not yet been tested against the full (undisclosed) test
  set.

  <section|Further tasks>

  <\itemize>
    <item>Evaluation of current solutions:

    <\itemize>
      <item>Evaluate a standard linear classifier as a baseline ?

      <item>Submit on Codalab a solution obtained with models attaining a low
      validation score
    </itemize>

    <item>New solutions:

    <\itemize>
      <item>Using a deep auto-encoder with pre-training (without layerwise
      pre-training it won't converge) but that can continue to evolve during
      the training to better fit the prediction cost

      <item>Adding to the cost a reconstruction penalty for the deep
      auto-encoder, trying to balance the two costs

      <item>Add to that a L1 penalty on the MLP weights

      <item>Think of new solutions
    </itemize>

    <item>Hyperparameter search
  </itemize>
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|2.1|1>>
    <associate|auto-4|<tuple|2.2|1>>
    <associate|auto-5|<tuple|2.3|2>>
    <associate|auto-6|<tuple|2.4|2>>
    <associate|auto-7|<tuple|3|3>>
    <associate|auto-8|<tuple|4|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Overview>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Models
      & implementation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>First idea
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Recurrent neural nets
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.3<space|2spc>MLP-generated linear
      classifier <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|2.4<space|2spc>MLP-generated MLP
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Current
      results> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>