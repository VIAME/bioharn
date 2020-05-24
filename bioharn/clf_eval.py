

def prototype_eval_clf():
    # hard coded prototype for classification evaluation
    import ubelt as ub
    config = {
        'deployed': ub.expandpath('$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip'),
    }
    from bioharn import clf_predict
    predictor = clf_predict.ClfPredictor(config)
    predictor.predict_sampler()
