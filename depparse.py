from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Callable, Iterator, Sequence, Text, Union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


@dataclass()
class Dep:
    """A word in a dependency tree.

    The fields are defined by https://universaldependencies.org/format.html.
    """
    id: Text
    form: Union[Text, None]
    lemma: Union[Text, None]
    upos: Text
    xpos: Union[Text, None]
    feats: Sequence[Text]
    head: Union[Text, None]
    deprel: Union[Text, None]
    deps: Sequence[Text]
    misc: Union[Text, None]


def read_conllu(path: Text) -> Iterator[Sequence[Dep]]:
    """Reads a CoNLL-U format file into sequences of Dep objects.

    The CoNLL-U format is described in detail here:
    https://universaldependencies.org/format.html
    A few key highlights:
    * Word lines contain 10 fields separated by tab characters.
    * Blank lines mark sentence boundaries.
    * Comment lines start with hash (#).

    Each word line will be converted into a Dep object, and the words in a
    sentence will be collected into a sequence (e.g., list).

    :return: An iterator over sentences, where each sentence is a sequence of
    words, and each word is represented by a Dep object.
    """

    with open(path, 'r') as fp:
        sentence = list()
        for line in fp:
            # remove \n
            line = line.strip()

            # empty line, process previous sentence
            if len(line) == 0:
                if len(sentence) > 0:
                    yield sentence
                    sentence = list()
                continue
            elif line.startswith('#'):
                continue

            # split, append token & tag
            parts = line.split('\t')
            # wrong line
            if len(parts) != 10:
                print('error: {}'.format(line))
                sentence = list()
                continue
            dep_id = parts[0] if parts[0] is not '_' else None
            # special case for _
            if parts[3] == 'SYM' and parts[1] == '_' and parts[2] == '_':
                form = '_'
                lemma = '_'
            else:
                form = parts[1] if parts[1] is not '_' else None
                lemma = parts[2] if parts[2] is not '_' else None
            upos = parts[3] if parts[3] is not '_' else None
            xpos = parts[4] if parts[4] is not '_' else None
            feats = [x.strip() for x in parts[5].split('|')] if parts[5] is not '_' else []
            head = parts[6] if parts[6] is not '_' else None
            deprel = parts[7] if parts[7] is not '_' else None
            deps = [x.strip() for x in parts[8].split('|')] if parts[8] is not '_' else []
            misc = parts[9] if parts[9] is not '_' else None
            new_dep = Dep(dep_id,
                          form,
                          lemma,
                          upos,
                          xpos,
                          feats,
                          head,
                          deprel,
                          deps,
                          misc,
                          )
            sentence.append(new_dep)


class Action(Enum):
    """An action in an "arc standard" transition-based parser."""
    SHIFT = 1
    LEFT_ARC = 2
    RIGHT_ARC = 3


def parse(deps: Sequence[Dep],
          get_action: Callable[[Sequence[Dep], Sequence[Dep]], Action]) -> None:
    """Parse the sentence based on "arc standard" transitions.

    Following the "arc standard" approach to transition-based parsing, this
    method creates a stack and a queue, where the input Deps start out on the
    queue, are moved to the stack by SHIFT actions, and are combined in
    head-dependent relations by LEFT_ARC and RIGHT_ARC actions.

    This method does not determine which actions to take; those are provided by
    the `get_action` argument to the method. That method will be called whenever
    the parser needs a new action, and then the parser will perform whatever
    action is returned. If `get_action` returns an invalid action (e.g., a
    SHIFT when the queue is empty), an arbitrary valid action will be taken
    instead.

    This method does not return anything; it modifies the `.head` field of the
    Dep objects that were passed as input. Each Dep object's `.head` field is
    assigned the value of its head's `.id` field, or "0" if the Dep object is
    the root.

    :param deps: The sentence, a sequence of Dep objects, each representing one
    of the words in the sentence.
    :param get_action: a function or other callable that takes the parser's
    current stack and queue as input, and returns an "arc standard" action.
    :return: Nothing; the `.head` fields of the input Dep objects are modified.
    """
    # init queue with given deps

    queue = deque()
    for dep in deps:
        try:
            int(dep.id)
        except:
            continue
        queue.append(dep)
    # init stack
    stack = list()

    # while loop to get action and parse the sentence
    while len(stack) > 1 or len(queue) > 0:
        # get action
        try:
            action = get_action(stack, queue)
        except Exception:
            break
        # shift the first word of queue to the stack
        if action == Action.SHIFT:
            if len(queue) > 0:
                dep = queue.popleft()
                stack.append(dep)
            # invalid action, use right_arc
            else:
                action = Action.RIGHT_ARC
        # left arc
        if action == Action.LEFT_ARC:
            if len(stack) < 2:
                break
            stack[-2].head = stack[-1].id
            stack.pop(-2)
        # right arc
        elif action == Action.RIGHT_ARC:
            if len(stack) < 2:
                break
            stack[-1].head = stack[-2].id
            stack.pop()

    # process stack
    while len(stack) >= 2:
        dep = stack.pop()
        dep.head = stack[-1].id

    # process the last word in the stack, set as root
    if len(stack) == 1:
        dep = stack.pop()
        dep.head = '0'


def get_feature_row(stack: Sequence[Dep], queue: Sequence[Dep]) -> dict:
    """
        given stack & queue, generate features.

    :param stack: current stack
    :param queue: queue for remaining words.
    :return: a dict of features.
    """

    feature_row = dict()
    if len(stack) >= 1:
        feature_row[f'stack_1_upos={stack[-1].upos}'] = 1
        feature_row[f'stack_1_xpos={stack[-1].xpos}'] = 1
        feature_row[f'stack_1_lemma={stack[-1].lemma.lower()}'] = 1
        feature_row[f'stack_1_form={stack[-1].form.lower()}'] = 1
        feature_row[f'stack_1_form_upos={stack[-1].form.lower() + stack[-1].upos}'] = 1
        feature_row[f'stack_1_lemma_upos={stack[-1].lemma.lower() + stack[-1].upos}'] = 1

    if len(stack) >= 2:
        feature_row[f'stack_2_upos={stack[-2].upos}'] = 1
        feature_row[f'stack_2_xpos={stack[-2].xpos}'] = 1
        feature_row[f'stack_2_lemma={stack[-2].lemma.lower()}'] = 1
        feature_row[f'stack_2_form={stack[-2].form.lower()}'] = 1
        feature_row[f'stack_2_form_upos={stack[-2].form.lower() + stack[-2].upos}'] = 1
        feature_row[f'stack_12_upos={stack[-1].upos + stack[-2].upos}'] = 1
        feature_row[f'stack_12_lemma={stack[-1].lemma + stack[-2].lemma}'] = 1
        feature_row[f'stack_12_form={stack[-1].form.lower() + stack[-2].form.lower()}'] = 1

    if len(stack) >= 3:
        feature_row[f'stack_3_upos={stack[-3].upos}'] = 1
        feature_row[f'stack_3_xpos={stack[-3].xpos}'] = 1
        feature_row[f'stack_3_lemma={stack[-3].lemma.lower()}'] = 1
        feature_row[f'stack_3_form={stack[-3].form.lower()}'] = 1

    if len(queue) >= 1:
        feature_row[f'queue_1_upos={queue[0].upos}'] = 1
        feature_row[f'queue_1_xpos={queue[0].xpos}'] = 1
        feature_row[f'queue_1_lemma={queue[0].lemma.lower()}'] = 1
        feature_row[f'queue_1_form={queue[0].form.lower()}'] = 1
        
    feature_row['stack_size'] = len(stack)
    feature_row['queue_size'] = len(queue)

    return feature_row


class Oracle:
    def __init__(self, deps: Sequence[Dep]):
        """Initializes an Oracle to be used for the given sentence.

        Minimally, it initializes a member variable `actions`, a list that
        will be updated every time `__call__` is called and a new action is
        generated.

        Note: a new Oracle object should be created for each sentence; an
        Oracle object should not be re-used for multiple sentences.

        :param deps: The sentence, a sequence of Dep objects, each representing
        one of the words in the sentence.
        """
        self.relations = set()

        for dep in deps:
            if dep.head is None:
                continue
            self.relations.add((dep.head, dep.id))
        # init actions to save all actions
        self.actions = list()
        # save features
        self.features = list()

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Returns the Oracle action for the given "arc standard" parser state.

        The oracle for an "arc standard" transition-based parser inspects the
        parser state and the reference parse (represented by the `.head` fields
        of the Dep objects) and:
        * Chooses LEFT_ARC if it produces a correct head-dependent relation
          given the reference parse and the current configuration.
        * Otherwise, chooses RIGHT_ARC if it produces a correct head-dependent
          relation given the reference parse and all of the dependents of the
          word at the top of the stack have already been assigned.
        * Otherwise, chooses SHIFT.

        The chosen action should be both:
        * Added to the `actions` member variable
        * Returned as the result of this method

        Note: this method should only be called on parser state based on the Dep
        objects that were passed to __init__; it should not be used for any
        other Dep objects.

        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken given the reference parse
        (the `.head` fields of the Dep objects).
        """
        # default action is shift
        action = Action.SHIFT
        # if stack is greater than 2
        if len(stack) >= 2:
            # if the top of the stack is the head of the word just below the top
            if (stack[-1].id, stack[-2].id) in self.relations:
                action = Action.LEFT_ARC
                self.relations.remove((stack[-1].id, stack[-2].id))
            else:
                # if top's dependents have not been processed completely, shift instead of right arc
                stack_top_is_head = False
                for dep_head, _ in self.relations:
                    if dep_head == stack[-1].id:
                        stack_top_is_head = True
                        break
                # if the top of the stack is the depend of the word just below the top
                if not stack_top_is_head and (stack[-2].id, stack[-1].id) in self.relations:
                    action = Action.RIGHT_ARC
                    self.relations.remove((stack[-2].id, stack[-1].id))

        # save action
        self.actions.append(action)

        # get features:
        feature_row = get_feature_row(stack, queue)
        self.features.append(feature_row)

        return action


class Classifier:
    def __init__(self, parses: Iterator[Sequence[Dep]]):
        """Trains a classifier on the given parses.

        There are no restrictions on what kind of classifier may be trained,
        but a typical approach would be to
        1. Define features based on the stack and queue of an "arc standard"
           transition-based parser (e.g., part-of-speech tags of the top words
           in the stack and queue).
        2. Apply `Oracle` and `parse` to each parse in the input to generate
           training examples of parser states and oracle actions. It may be
           helpful to modify `Oracle` to call the feature extraction function
           defined in 1, and store the features alongside the actions list that
           `Oracle` is already creating.
        3. Train a machine learning model (e.g., logistic regression) on the
           resulting features and labels (actions).

        :param parses: An iterator over sentences, where each sentence is a
        sequence of words, and each word is represented by a Dep object.
        """

        transition_features = list()
        transition_labels = list()
        for sentence in parses:
            oracle = Oracle(sentence)
            parse(sentence, oracle)
            transition_features.extend(oracle.features)
            transition_labels.extend(oracle.actions)

        # used to map features into array
        self.dict_vectorizer = DictVectorizer(sparse=True)
        # convert features to numpy array
        feature_matrix = self.dict_vectorizer.fit_transform(transition_features)

        # used to convert label names into array
        self.label_encoder = LabelEncoder()

        # convert label into array
        label_vector = self.label_encoder.fit_transform([action.value for action in transition_labels])

        # logistic regression classifier
        self.classifier = LogisticRegression(random_state=0, solver='lbfgs', penalty='l2', class_weight='balanced', max_iter=100, multi_class='multinomial')

        # train model
        self.classifier.fit(X=feature_matrix, y=label_vector)

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Predicts an action for the given "arc standard" parser state.

        There are no restrictions on how this prediction may be made, but a
        typical approach would be to convert the parser state into features,
        and then use the machine learning model (trained in `__init__`) to make
        the prediction.

        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken.
        """

        feature_row = get_feature_row(stack, queue)
        # convert features to numpy array
        feature_matrix = self.dict_vectorizer.transform(feature_row)

        pred_action_idx = self.classifier.predict(feature_matrix)
        pred_action_value = self.label_encoder.inverse_transform(pred_action_idx)[0]
        pred_action = Action(pred_action_value)

        return pred_action

