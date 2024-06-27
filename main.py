from typing import List, Tuple

import nltk, inspect, sys, hashlib

from nltk.corpus import brown

import math

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist, LidstoneProbDist

from nltk.tag import map_tag

from adrive2 import trim_and_warn

assert (
    map_tag("brown", "universal", "NR-TL") == "NOUN"
), """
Brown-to-Universal POS tag map is out of date."""


class HMM:
    def __init__(self, train_data: List[List[Tuple[str, str]]]) -> None:
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        """
        self.train_data = train_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []
        self.viterbi = []
        self.backpointer = []
        self.start_tag = '<s>'
        self.end_tag = '</s>'

    # Q1.1

    # Compute emission model using ConditionalProbDist with
    # a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with
    #    +0.001 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data: List[List[Tuple[str, str]]]) -> Tuple[ConditionalProbDist, List[str]]:
        """
        Compute an emission model based on labelled training data.
        Don't forget to lowercase the observation otherwise it mismatches
        the test data.

        :param train_data: The training dataset, a list of sentences with tags
        :return: The emission probability distribution and a list of the states
        """
        # TODO prepare data

        # Don't forget to lowercase the observation,
        # otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
        data = []
        states = []
        for tagged_sent in train_data:
            data += [(t, w.lower()) for w, t in tagged_sent]
            # collect the states
            for _, t in tagged_sent:
                if t not in states:
                    states.append(t)

        # TODO compute the emission model
        emission_FD = ConditionalFreqDist(data)

        probdist_func = lambda freqdist: LidstoneProbDist(freqdist=freqdist, gamma=0.001, bins=freqdist.B() + 1)
        self.emission_PD = ConditionalProbDist(cfdist=emission_FD, probdist_factory=probdist_func)

        # Sorting helps with consistent tie breaking in Viterbi.
        self.states = sorted(states)

        return self.emission_PD, self.states

    # Q1.1

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self, state: str, word: str) -> float:
        """
        The log of the estimated probability of emitting a word from a state.

        If you use the math library to compute a log base 2, make sure to
        use math.log(p,2) (rather than math.log2(p))

        :param state: the state name
        :param word: the word
        :return: log base 2 of the estimated emission probability
        """
        return math.log(self.emission_PD[state].prob(word), 2)

    # Q1.2
    # Compute transition model using ConditionalProbDist with the same
    #  estimator as above (but without the extra bin)
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data: List[List[Tuple[str, str]]]) -> ConditionalProbDist:
        """
        Compute a transition model using a ConditionalProbDist based on
          labelled data.

        :param train_data: The training dataset, a list of sentences with tags
        :return: The transition probability distribution
        """
        # TODO: prepare the data
        data = []
        tags = []

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL <s> and the END SYMBOL </s>
        for s in train_data:
            sent_tags = [self.start_tag]
            sent_tags += [t for _, t in s]
            sent_tags.append(self.end_tag)
            tags += sent_tags

        for i in range(len(tags) - 1):
            data.append((tags[i], tags[i + 1]))

        # TODO compute the transition model
        transition_FD = ConditionalFreqDist(data)

        probdist_func = lambda freqdist: LidstoneProbDist(freqdist=freqdist, gamma=0.001, bins=freqdist.B())
        self.transition_PD = ConditionalProbDist(cfdist=transition_FD, probdist_factory=probdist_func)

        return self.transition_PD

    # Q1.2
    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self, state1: str, state2: str) -> float:
        """
        The log of the estimated probability of a transition from one state to another

        If you use the math library to compute a log base 2, make sure to
        use math.log(p,2) (rather than math.log2(p))

        :param state1: the first state name
        :param state2: the second state name
        :return: log base 2 of the estimated transition probability
        """
        return math.log(self.transition_PD[state1].prob(state2), 2)

    # Train the HMM
    def train(self) -> None:
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part 2: Implementing the Viterbi algorithm.

    # Q2.1
    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag and the total number of observations.
    def initialise(self, observation: str, number_of_observations: int) -> None:
        """
        Initialise data structures self.viterbi and self.backpointer for
        tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :param number_of_observations: the number of observations
        """
        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (- log-base-2 probabilities)
        # TODO

        # Reset viterbi data structure
        self.viterbi = []

        # We build the viterbi data structure by going through the states (found tags in training) and
        # determine the probability of the first word in that state
        for state in self.states:
            # transition probability for this state
            trans_prob = -self.tlprob(self.start_tag, state)

            # emission probability
            em_prob = -self.elprob(state, observation)

            # We use a list as we will add values later
            self.viterbi.append([trans_prob + em_prob])

        # Initialise step 0 of backpointer (backpointer data structure)
        # for every state, we initialise a list of 0, which we will change
        # as soon as we update the backpointers to traverse the viterbi table
        # TODO
        self.backpointer = [[0] for _ in self.states]

    # Q2.1
    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42
    def get_viterbi_value(self, state: str, step: int) -> float:
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :return: The value (a cost) for state as of step
        """
        return self.viterbi[self.states.index(state)][step]

    # Q2.1
    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state: str, step: int) -> str:
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :return: The state name to go back to at step-1
        """
        return self.states[self.backpointer[self.states.index(state)][step]]

    # Q2.2
    # Tag a new sentence using the trained model and already initialised
    # data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer data structures.
    # Describe your implementation with comments.
    def tag(self, observations: List[str]) -> List[str]:
        """
        Tag a new sentence using the trained model and already initialised
        data structures.

        :param observations: List of words (a sentence) to be tagged
        :return: List of tags corresponding to each word of the input
        """
        tags = []

        for t in range(1, len(observations)):  # iterate over steps
            for s in range(len(self.states)):  # iterate over states
                # update the viterbi and backpointer data structures
                current_state = self.states[s]

                # We save probability of getting from a prev state to current state
                # by going through all states and calculating previous prob
                current_state_probabilities = []
                for i in range(len(self.states)):
                    # (we use costs, not probabilities)
                    transition_prob = self.viterbi[i][t - 1] - self.tlprob(self.states[i], current_state)
                    emission_prob = -self.elprob(current_state, observations[t])
                    current_state_probabilities.append(transition_prob + emission_prob)

                # min cost = max prob
                min_cost = min(current_state_probabilities)

                # If the value of the backpointer is not uniquely determined
                # because there are two options with the same cost,
                # break the tie by storing the backpointer to the first state
                # (according to the order of self.states)
                min_cost_index = current_state_probabilities.index(min_cost)

                # Update viterbi and backpointer data structures
                self.viterbi[s].append(min_cost)
                self.backpointer[s].append(min_cost_index)

        # TODO
        # Add a termination step with cost based solely on
        #   cost of transition to </s> , end of sentence.
        termination = []
        last_ob_index = len(observations) - 1
        for s in range(len(self.states)):
            previous_prob = self.viterbi[s][last_ob_index]
            transition_prob = -self.tlprob(self.states[s], self.end_tag)
            termination.append(previous_prob + transition_prob)

        # TODO
        # Reconstruct the tag sequence using the backpointers.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.
        last_backpointer = termination.index(min(termination))
        tags.append(self.states[last_backpointer])  # fixme

        for t in range(len(observations) - 1, 0, -1):
            current_backpointer = self.backpointer[last_backpointer][t]
            tags.append(self.states[current_backpointer])
            last_backpointer = current_backpointer

        tags.reverse()
        return tags

    def tag_sentence(self, sentence: List[str]) -> List[str]:
        """
        Initialise the HMM, lower case and tag a sentence. Returns a list of tags.
        :param sentence: the sentence
        """
        normalised_sent = [w.lower() for w in sentence]
        observation_num = len(normalised_sent)
        self.initialise(normalised_sent[0], observation_num)

        # print('ADJ -> gaudy:\t', self.elprob('ADJ', 'gaudy'))
        # print('ADV -> gaudy:\t', self.elprob('ADV', 'gaudy'))
        # print('VERB -> ADJ:\t', self.tlprob('VERB', 'ADJ'))
        # print('VERB -> ADV:\t', self.tlprob('VERB', 'ADV'))

        return self.tag(normalised_sent)


def answer_question_2_3() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :return: incorrectly tagged sequence, correctly tagged sequence and
    your answer [max 75 words not including the sequences]
    """

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand

    tagged_sequence = [
        ('``', '.'),
        ('My', 'DET'),
        ('taste', 'NOUN'),
        ('is', 'VERB'),
        ('gaudy', 'ADV'),
        ('.', '.'),
    ]

    correct_sequence = [
        ('``', '.'),
        ('My', 'DET'),
        ('taste', 'NOUN'),
        ('is', 'VERB'),
        ('gaudy', 'ADJ'),
        ('.', '.'),
    ]

    # ADJ -> gaudy:    -22.497055589257112
    # ADV -> gaudy:    -21.4723036174209
    # VERB -> ADJ:     -4.349754167251695
    # VERB -> ADV:     -3.8122000376088043

    # Why do you think the tagger tagged this example incorrectly?
    answer = inspect.cleandoc(
        """
    The sequential transition probability only considers the previous word so it can't
    reflect that "gaudy" is dependent on "taste". It chooses the more likely transition from
    VERB -> ADV (-3.81 < -4.35 [VERB -> ADJ]), supported by the higher emission probability
    (P(ADV|gaudy)=-21.47 < P(ADV|gaudy)=-22.50).
    """
    )

    return tagged_sequence, correct_sequence, answer


# Q3.1
def hard_em(
    labeled_data: List[List[Tuple[str, str]]],
    unlabeled_data: List[List[str]],
    k: int,
) -> HMM:
    """
    Run k iterations of hard EM on the labeled and unlabeled data.
    Follow the pseudo-code in the coursework instructions.

    :param labeled_data:
    :param unlabeled_data:
    :param k: number of iterations
    :return: HMM model trained with hard EM.
    """
    T_0 = T_i = HMM(labeled_data)
    T_0.train()

    for _ in range(k):
        P_i1 = []
        for s in unlabeled_data:
            P_i1.append(list(zip(s, T_i.tag_sentence(s))))
        T_i = HMM(labeled_data + P_i1)
        T_i.train()

    return T_i


# Q3.2
def answer_question3_2():
    """
    Sentence:  In    fact  he    seemed   delighted  to  get   rid  of  them   .
    Gold POS:  ADP   NOUN  PRON  VERB     VERB      PRT  VERB  ADJ  ADP  PRON  .
    T_0     :  PRON  VERB  NUM    ADP     ADJ       PRT  VERB  NUM  ADP  PRON  .
    T_3     :  PRON  VERB  PRON  VERB     ADJ       PRT  VERB  NUM  ADP  NOUN  .

    1) T_0 erroneously tagged "he" as "NUM" and T_3 correctly identifies it as "PRON".
        Speculate why additional unlabeled data might have helped in that case.
        Refer to the training data (inspect the 20 sentences!).
    2) Where does T_3 mislabel a word but T_0 is correct? Why do you think hard
       EM went wrong in that case?

    :rtype: str
    :return: your answer [max 150 words]
    """

    return inspect.cleandoc(
        """
        1)There are other pronouns in the training set. "he" appears in the same context as
        those in the unlabeled data which might be why it helped tagging "he" as PRON.

        2)This might be because of the effects of semi-supervised learning and using
        pseudo-labelled data as training data for later iterations, leading to, along with smoothing,
        data pollution and incorrect tagging.
    """
    )


def answer_question_4_1() -> str:
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 100 words]
    """
    return inspect.cleandoc(
        """
    Data sparsity: the upenn_tagset is a much larger tagset (len=43) than the universal tagset (12) which means that the
    model will have to deal with an increased data sparsity issue due to more states.
    With a small training set, this is likely to lead to a dropped accuracy and worse performance.
    """
    )


def answer_question_4_2() -> str:
    """
    Suppose you have a hand-crafted probabilistic context-free grammar
    and lexicon that has 100% coverage on constructions but less than
    100% lexical coverage. How could you use a pre-trained POS tagger
    to ensure that the grammar produces a parse for any well-formed
    sentence, even when the lexicon doesn’t contain some of the words
    within that sentence?

    Will your approach always do as well as or better than the original
    parser on its own?  Why or why not?

    :rtype: str
    :return: your answer [max 150 words]
    """

    return inspect.cleandoc(
        """
    You could use a pre-trained POS tagger to provide tags for words which are
    not covered by the lexicon, and modify the grammar to include a rule for
    generating based on a tag, and perhaps a probability associated with it.
    That probability could be derived from the sequential context of the tag,
    and of the new word.

    This approach can do better than the original parser by allowing to parse
    sentences that did not have lexical coverage before. However, the performance
    is dependent on the POS-tagger accuracy.
    """
    )


def answer_question_4_3() -> str:
    """
    Essay question on noisy channel model; see coursework instructions.

    :rtype: str
    :return: your answer [max 400 words]
    """

    return inspect.cleandoc(
        """
    We identify 4 noisy channel components: source, channel, noise, goal.

    For the translation, the source is the Russian text that needs translation.
    The channel is the used translation method. The noise are various factors that
    can introduce errors durign the translation process, e.g. human-bound errors.
    The goal is the English translation produced by the channel.

    For the tagging, the source is the original text that needs to be tagged.
    The channel is the model (or method) used for tagging.
    The noise are various factors that can introduce errors,
    e.g. assumptions that models make such as HMMs only based on the previous word
    instead of the entire sequence.
    The goal is the output of the channel, i.e. the tagged sentence.

    Both problems have in common that they require access to language resources
    to achieve their goal.
    
    E.g. for translation, we need access to translation lexica (provides translations for words) which could be seen as a pendant to
    annotated tagging data ('translates' word to tag in a given context). That context is
    often also prevalent in translation lexica (e.g. different translations if noun or verb for same word).
    Tags can be relevant for translation.

    The difference in those resources is that they need to provide semantically different information:
    tagging and translation are, afterall, different tasks eventhough both language-based.

    Those resources can be used for training and improving the translation and tagging systems, and evaluating their output.

    I would say translation is the more difficult task as it usually requires to
    have the correct tag for a given word to obtain its correct translation, which means
    it subsumes the tagging problem as well.
    """
    )


def compute_acc(hmm, test_data, print_mistakes):
    """
    Computes accuracy (0.0 - 1.0) of model on some data.
    :param hmm: the HMM
    :type hmm: HMM
    :param test_data: the data to compute accuracy on.
    :type test_data: list(list(tuple(str, str)))
    :param print_mistakes: whether to print the first 10 model mistakes
    :type print_mistakes: bool
    :return: float
    """
    # TODO: modify this to print the first 10 sentences with at
    #  least one mistake if print_mistakes = True
    correct = 0
    incorrect = 0
    mistakes = []
    for sentence in test_data:
        s = [word for (word, tag) in sentence]
        tags = hmm.tag_sentence(s)

        for ((word, gold), tag) in zip(sentence, tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1
        mistakes.append(zip(sentence, tags))

    if print_mistakes:
        for i in range(1, 11):
            print(f'Mistake {i}:')
            print(list(mistakes[i]))
            print()

    return float(correct) / (correct + incorrect)


# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def answers():
    global tagged_sentences_universal, test_data_universal, train_data_universal, model, test_size, train_size, ttags, correct, incorrect, accuracy, good_tags, bad_tags, answer2_3, answer5, answer4_2, answer4_1, answer3_2, answer4_3, t0_acc, tk_acc

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories="news", tagset="universal")

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - test_size

    # tail test set
    test_data_universal = tagged_sentences_universal[-test_size:]  # [:test_size]
    train_data_universal = tagged_sentences_universal[:train_size]  # [test_size:]
    if (
        hashlib.md5(
            "".join(
                map(
                    lambda x: x[0],
                    train_data_universal[0] + train_data_universal[-1] + test_data_universal[0] + test_data_universal[-1],
                )
            ).encode("utf-8")
        ).hexdigest()
        != "164179b8e679e96b2d7ff7d360b75735"
    ):
        print(
            "!!!test/train split (%s/%s) incorrect -- this should not happen, please contact a TA !!!" % (len(train_data_universal), len(test_data_universal)),
            file=sys.stderr,
        )

    # Create instance of HMM class and initialise the training set.
    model = HMM(train_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample = model.elprob("VERB", "is")
    if not (type(e_sample) == float and e_sample <= 0.0):
        print(
            "elprob value (%s) must be a log probability" % e_sample,
            file=sys.stderr,
        )

    t_sample = model.tlprob("VERB", "VERB")
    if not (type(t_sample) == float and t_sample <= 0.0):
        print(
            "tlprob value (%s) must be a log probability" % t_sample,
            file=sys.stderr,
        )

    if not (type(model.states) == list and len(model.states) > 0 and type(model.states[0]) == str):
        print(
            "model.states value (%s) must be a non-empty list of strings" % model.states,
            file=sys.stderr,
        )
    else:
        print("states: %s\n" % model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s = "the cat in the hat came back".split()
    ttags = model.tag_sentence(s)
    print("Tagged a trial sentence:\n  %s" % list(zip(s, ttags)))

    v_sample = model.get_viterbi_value("VERB", 5)
    if not (type(v_sample) == float and 0.0 <= v_sample):
        print("viterbi value (%s) must be a cost" % v_sample, file=sys.stderr)

    b_sample = model.get_backpointer_value("VERB", 5)
    if not (type(b_sample) == str and b_sample in model.states):
        print(
            "backpointer value (%s) must be a state name" % b_sample,
            file=sys.stderr,
        )

    # check the model's accuracy (% correct) using the test set
    accuracy = compute_acc(model, test_data_universal, print_mistakes=True)
    print("\nTagging accuracy for test set of %s sentences: %.4f" % (test_size, accuracy))

    # Tag the sentence again to put the results in memory for automarker.
    model.tag_sentence(s)

    # Question 3.1
    # Set aside the first 20 sentences of the training set
    num_sentences = 20
    semi_supervised_labeled = train_data_universal[:num_sentences]  # type list(list(tuple(str, str)))
    semi_supervised_unlabeled = [[word for (word, tag) in sent] for sent in train_data_universal[num_sentences:]]  # type list(list(str))
    print("Running hard EM for Q3.2. This may take a while...")
    t0 = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 0)  # 0 iterations
    tk = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 3)
    print("done.")

    t0_acc = compute_acc(t0, test_data_universal, print_mistakes=False)
    tk_acc = compute_acc(tk, test_data_universal, print_mistakes=False)
    print("\nTagging accuracy of T_0: %.4f" % (t0_acc))
    print("\nTagging accuracy of T_3: %.4f" % (tk_acc))
    ########

    # Print answers
    bad_tags, good_tags, answer2_3 = answer_question_2_3()
    print("\nA tagged-by-your-model version of a sentence:")
    print(bad_tags)
    print("The tagged version of this sentence from the corpus:")
    print(good_tags)
    print("\nDiscussion of the difference:")
    print(answer2_3)
    answer3_2 = answer_question3_2()
    print("\nFor Q3.2:")
    print(answer3_2)
    answer4_1 = answer_question_4_1()
    print("\nFor Q4.1:")
    print(answer4_1)
    answer4_2 = answer_question_4_2()
    print("\nFor Q4.2:")
    print(answer4_2)
    answer4_3 = answer_question_4_3()
    print("\nFor Q4.3:")
    print(answer4_3)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--answers":
        import adrive2
        from autodrive_embed import run, carefulBind

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive2.a2answers, errlog)
    else:
        answers()
