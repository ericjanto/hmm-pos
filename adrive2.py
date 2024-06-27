import sys,traceback

states=['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X','bogus']

def trim_and_warn(name, max_len, s):
    if len(s) > max_len:
        print("\nWarning - truncating output of %s: your answer has %i characters but the limit is %i" % (
        name, len(s), max_len), file=sys.stderr)
    return s[:max_len]

def check_viterbi(model,n):
  return check_mod_prop(model.get_viterbi_value,n)

def check_mod_prop(getfn,n):
  global states
  try:
    ovflowed=getfn("VERB",n+1) is None
  except:
    ovflowed=True
  statesCheck=True
  for s in states:
    try:
      v=getfn(s,n-1)
    except:
      if s!='bogus':
        statesCheck=False
        break
  return (ovflowed and getfn("VERB",0) is not None and getfn("VERB",n) is not None,
          statesCheck)

def check_bp(model,n):
  return check_mod_prop(model.get_backpointer_value,n)

from collections.abc import KeysView

def a2answers(gdict,errlog):
  globals().update(gdict)
  errs=0
  (cerrs,ans)=carefulBind(
    [('a1_1aa','list(model.states) if isinstance(model.states,KeysView) else model.states'),
     ('a1_1b','len(model.emission_PD["VERB"].samples()) if type(model.emission_PD)==nltk.probability.ConditionalProbDist else FAILED'),
     ('a1_1c','-model.elprob("VERB","attack")'),
     ('a1_1d','model.emission_PD._probdist_factory.__class__.__name__ if model.emission_PD is not None else FAILED'),
     ('a1_2a','len(model.transition_PD["VERB"].samples()) if type(model.transition_PD)==nltk.probability.ConditionalProbDist else FAILED'),
     ('a1_2b','-model.tlprob("VERB","DET")'),
     ('a2_1a', 'model.get_viterbi_value("DET",0)'),
     ('a2_1b', 'model.get_backpointer_value("DET",1)'),
     ('a2_2a','accuracy'),
     ('a2_2b','model.get_viterbi_value("VERB",5)'),
     ('a2_2c','min((model.get_viterbi_value(s,-1) for s in model.states)) if len(model.states)>0 else FAILED'),
     ('a2_2d','list(ttags)'),
     ('a2_3_good_tags', 'bad_tags'),
     ('a2_3_bad_tags', 'good_tags'),
     ('a2_3c', 'answer2_3'),
     ("a3_1_t0", "t0_acc"),
     ("a3_1_tk", "tk_acc"),
     ("a3_2", "answer3_2"),
     ('a4_1','answer4_1'),
     ('a4_2','answer4_2'),
     ('a4_3','answer4_3'),
     ],globals(),errlog)
  errs+=cerrs
  s = 'they model the world'.split()
  try:
    ttags = model.tag_sentence(s)
    gdict["viterbi_matrix"] = [[None for _ in model.states] for _ in range(len(s))]
    gdict["bp_matrix"] = [[None for _ in model.states] for _ in range(len(s) - 1)]
    for position in range(len(s)):
        for i, state in enumerate(sorted(model.states)):
            try:
                gdict["viterbi_matrix"][position][i] = model.get_viterbi_value(state, position)
                if position > 0:
                    gdict["bp_matrix"][position-1][i] = model.get_backpointer_value(state, position)
            except NotImplementedError:
                pass
            except Exception as e:
                errs += 1
                print("Exception in computing viterbi/backpointer for %s at step %i:\n%s" % (state, position, repr(e)),
                      file=errlog)
                traceback.print_tb(sys.exc_info()[2], None, errlog)

  except NotImplementedError:
    pass
  except Exception as e:
    errs+=1
    print("Exception in initialising model in adrive2:\n%s"%repr(e),
          file=errlog)
    traceback.print_tb(sys.exc_info()[2],None,errlog)
  (cerrs,nans)=carefulBind(
    [("a2_2full_vit", "viterbi_matrix"),
      ("a2_2full_bp", "bp_matrix")],gdict,errlog)
  ans.update(nans)
  errs+=cerrs
  return (ans,errs)

if __name__ == '__main__':
  from autodrive import run, answers, HMM, carefulBind
  with open("userErrs.txt","w") as errlog:
    run(answers,a2answers,errlog)
