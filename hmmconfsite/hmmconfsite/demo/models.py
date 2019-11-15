import os, tempfile
from django.db import models
from django.dispatch import receiver
from django.conf import settings
from sorl.thumbnail import ImageField
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import make_logger
from . import utils
from hmmconf import visualize
from pm4py.objects.petri.importer import factory as pnml_importer


def get_file_barplot_logfwd_fp(instance, filename):
    parent_dir = os.path.join('images', 'barplot', 'logfwd')
    return os.path.join(parent_dir, filename)


def get_file_barplot_logfwd_before_obs_fp(instance, filename):
    parent_dir = os.path.join('images', 'barplot', 'logfwd_before_obs')
    return os.path.join(parent_dir, filename)


def get_file_net_logfwd_fp(instance, filename):
    parent_dir = os.path.join('images', 'net', 'logfwd')
    return os.path.join(parent_dir, filename)


def get_file_net_logfwd_before_obs_fp(instance, filename):
    parent_dir = os.path.join('images', 'net', 'logfwd_before_obs')
    return os.path.join(parent_dir, filename)


class Log(models.Model):
    logger = make_logger('Log')

    name = models.CharField(max_length=500)
    file_pnml = models.FileField(max_length=500)
    file_logstartprob = models.FileField(max_length=500)
    file_logtranscube = models.FileField(max_length=500)
    file_logtranscube_d = models.FileField(max_length=500)
    file_logemitmat = models.FileField(max_length=500)
    file_logemitmat_d = models.FileField(max_length=500)
    file_confmat = models.FileField(max_length=500)


class State(models.Model):
    log = models.ForeignKey(Log, on_delete=models.CASCADE)
    state = models.CharField(max_length=500)
    state_id = models.IntegerField()

    def __repr__(self):
        repr_ = '{}({}, {})'
        repr_ = repr_.format(State.__name__,
                             self.state, self.state_id)
        return repr_


def make_neglogfwd_barplot(logfwd, highlight_color='lawngreen'):
    state_list = State.objects.values_list('state', flat=True).order_by('state_id')
    df = pd.DataFrame({
        'state': state_list,
        'logprob': logfwd,
    }).replace([-np.inf], -1e8)
    df['neg_logprob'] = -df['logprob']

    fig, ax = plt.subplots()
    min_negprob = df['neg_logprob'].min()
    clrs = ['grey' if x > min_negprob else highlight_color for x in df['neg_logprob'].values]
    sns.barplot(x='state', y='neg_logprob', data=df, ax=ax, palette=clrs)

    # rotate xtick labels
    for item in ax.get_xticklabels():
        item.set_rotation(45)

    return fig, ax


def read_pnml():
    log = Log.objects.first()
    net_fp = log.file_pnml.path
    net, init_marking, final_marking = pnml_importer.apply(net_fp)
    return net, init_marking, final_marking


def make_highlighted_net(logfwd, highlight_color='greenyellow'):
    net, init_marking, final_marking = read_pnml()
    state_list = State.objects.values_list('state', flat=True).order_by('state_id')
    highlight_state_id = np.argmax(logfwd)
    highlight_state = state_list[highlight_state_id]

    graph_attr = {
        'background': 'white',
        'rankdir': 'LR',
    }

    dot_g = visualize.petrinet2dot(
        net, init_marking, final_marking, 
        highlight_state, highlight_color=highlight_color,
        graph_attr=graph_attr
    )

    return dot_g


class Event(models.Model):
    logger = make_logger('Event')

    log = models.ForeignKey(Log, on_delete=models.CASCADE)
    caseid = models.CharField(max_length=120)
    index = models.IntegerField()
    activity = models.CharField(max_length=500)
    activity_id = models.IntegerField()
    mode_state = models.ForeignKey(
        State, related_name='mode_state', on_delete=models.CASCADE
    )
    mode_state_before_obs = models.ForeignKey(
        State, null=True, related_name='mode_state_before_obs', on_delete=models.CASCADE
    )
    is_train = models.BooleanField()

    logfwd_str = models.CharField(max_length=1000)
    logfwd_before_obs_str = models.CharField(max_length=1000)
    stateconf = models.FloatField(default=-1.0)
    emitconf = models.FloatField(default=-1.0)
    finalconf = models.FloatField(default=-1.0)
    injected_distance = models.FloatField(default=-1.0)
    completeness = models.FloatField(default=-1.0)

    file_barplot_logfwd = ImageField(
        max_length=500,
        upload_to=get_file_barplot_logfwd_fp, 
        blank=True
    )
    file_barplot_logfwd_before_obs = ImageField(
        max_length=500,
        upload_to=get_file_barplot_logfwd_before_obs_fp, 
        blank=True
    )
    file_net_logfwd = ImageField(
        max_length=500,
        upload_to=get_file_net_logfwd_fp, 
        blank=True
    )
    file_net_logfwd_before_obs = ImageField(
        max_length=500,
        upload_to=get_file_net_logfwd_before_obs_fp, 
        blank=True
    )

    def __repr__(self):
        repr_ = '{}({}, {}, {})'
        repr_ = repr_.format(Event.__name__, self.caseid, self.index, self.activity)
        return repr_

    def get_file_barplot_logfwd(self):
        if bool(self.file_barplot_logfwd) is True:
            return self.file_barplot_logfwd

        # create the negative log forward probability bar plot
        logfwd = utils.arr_str_to_npy_arr(self.logfwd_str)
        fig, ax = make_neglogfwd_barplot(logfwd)
        fig = ax.get_figure()

        fname = '{}-caseid_{}-event_{}-barplot_neglogfwd.png'
        fname = fname.format(self.log.name, self.caseid, self.index)

        # make temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            png_fp = os.path.join(tempdir, fname)
            fig.savefig(png_fp, bbox_inches='tight')
            plt.close()

            # move figure to chart directory
            with open(png_fp, 'rb') as f:
                self.file_barplot_logfwd.save(fname, File(f), save=True)
                self.save()

        return self.file_barplot_logfwd

    def get_file_barplot_logfwd_before_obs(self):
        if bool(self.file_barplot_logfwd_before_obs) is True:
            return self.file_barplot_logfwd_before_obs

        # create the negative log forward probability bar plot
        logfwd = utils.arr_str_to_npy_arr(self.logfwd_before_obs_str)
        fig, ax = make_neglogfwd_barplot(logfwd)
        fig = ax.get_figure()

        fname = '{}-caseid_{}-event_{}-barplot_neglogfwd_before_obs.png'
        fname = fname.format(self.log.name, self.caseid, self.index)

        # make temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            png_fp = os.path.join(tempdir, fname)
            fig.savefig(png_fp, bbox_inches='tight')
            plt.close()

            # move figure to chart directory
            with open(png_fp, 'rb') as f:
                self.file_barplot_logfwd_before_obs.save(fname, File(f), save=True)
                self.save()

        return self.file_barplot_logfwd_before_obs

    def get_file_net_logfwd(self):
        if bool(self.file_net_logfwd) is True:
            return self.file_net_logfwd

        # create the highlighted net
        logfwd = utils.arr_str_to_npy_arr(self.logfwd_str)
        dot_g = make_highlighted_net(logfwd)

        # make temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            fname = '{}-caseid_{}-event_{}-net_logfwd'
            fname = fname.format(self.log.name, self.caseid, self.index)
            png_fname = '{}.png'.format(fname)

            dot_g.render(fname, tempdir, format='png', cleanup=True)
            png_fp = os.path.join(tempdir, png_fname)

            with open(png_fp, 'rb') as f:
                self.file_net_logfwd.save(png_fname, File(f), save=True)
                self.save()

        return self.file_net_logfwd

    def get_file_net_logfwd_before_obs(self):
        if bool(self.file_net_logfwd_before_obs) is True:
            return self.file_net_logfwd_before_obs

        # create the highlighted net
        logfwd = utils.arr_str_to_npy_arr(self.logfwd_before_obs_str)
        dot_g = make_highlighted_net(logfwd)

        # make temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            fname = '{}-caseid_{}-event_{}-net_logfwd_before_obs'
            fname = fname.format(self.log.name, self.caseid, self.index)
            png_fname = '{}.png'.format(fname)

            dot_g.render(fname, tempdir, format='png', cleanup=True)
            png_fp = os.path.join(tempdir, png_fname)

            with open(png_fp, 'rb') as f:
                self.file_net_logfwd_before_obs.save(png_fname, File(f), save=True)
                self.save()

        return self.file_net_logfwd_before_obs


def create_objects(file_pnml, file_logstartprob, 
                   file_logtranscube, file_logtranscube_d,
                   file_logemitmat, file_logemitmat_d, 
                   file_confmat, file_state_csv, file_event_csv, row_limit=10000):
    # read csv
    df = pd.read_csv(file_event_csv.temporary_file_path())
    log_name = df['log']
    log = Log.objects.create(
        name=log_name, file_pnml=file_pnml,
        file_logstartprob=file_logstartprob,
        file_logtranscube=file_logtranscube,
        file_logtranscube_d=file_logtranscube_d,
        file_logemitmat=file_logemitmat,
        file_logemitmat_d=file_logemitmat_d,
        file_confmat=file_confmat
    )

    row_limit -= 1

    # states
    state_df = pd.read_csv(file_state_csv.temporary_file_path())
    state_df = state_df.sort_values('state_id')
    for row in state_df[['state_id', 'state']].itertuples():
        state_id = row.state_id
        state = row.state
        State.objects.create(log=log, state_id=state_id, state=state)
        row_limit -= 1

    grouped = df.groupby('caseid', sort=True)
    for caseid, group in grouped:
        n_rows = group.shape[0]

        # do not add the case if row limit will be passed
        if row_limit < n_rows:
            break
        row_limit -= n_rows

        # add events
        index = 0
        for row in group.itertuples():
            activity = row.activity
            activity_id = row.activityid
            logfwd_str = row.logfwd
            logfwd_before_obs_str = row.logfwd_before_obs
            finalconf = row.finalconf
            injected_distance = row.injected_distance
            completeness = row.completeness
            mode_state = State.objects.filter(
                state_id__exact=row.mode_state_id
            )[0]
            mode_state_before_obs = None
            if row.mode_state_before_obs_id >= 0:
                mode_state_before_obs = State.objects.filter(
                    state_id__exact=row.mode_state_before_obs_id
                )[0]
            is_train = row.is_train
            stateconf = row.stateconf
            emitconf = row.emitconf

            Event.objects.create(
                log=log, caseid=caseid,
                is_train=is_train,
                index=index, activity=activity,
                activity_id=activity_id,
                mode_state=mode_state,
                mode_state_before_obs=mode_state_before_obs,
                logfwd_str=logfwd_str,
                logfwd_before_obs_str=logfwd_before_obs_str,
                stateconf=stateconf, emitconf=emitconf,
                finalconf=finalconf,
                injected_distance=injected_distance,
                completeness=completeness
            )

            index += 1


@receiver(models.signals.post_delete, sender=Log)
def auto_delete_file_on_delete_log(sender, instance, **kwargs):
    instance.file_pnml.delete(save=False)
    instance.file_logstartprob.delete(save=False)
    instance.file_logtranscube.delete(save=False)
    instance.file_logtranscube_d.delete(save=False)
    instance.file_logemitmat.delete(save=False)
    instance.file_logemitmat_d.delete(save=False)
    instance.file_confmat.delete(save=False)


@receiver(models.signals.post_delete, sender=Event)
def auto_delete_file_on_delete_event(sender, instance, **kwargs):
    if bool(instance.file_barplot_logfwd):
        instance.file_barplot_logfwd.delete(save=False)
    if bool(instance.file_barplot_logfwd_before_obs):
        instance.file_barplot_logfwd_before_obs.delete(save=False)
    if bool(instance.file_net_logfwd):
        instance.file_net_logfwd(save=False)
    if bool(instance.file_net_logfwd_before_obs):
        instance.file_net_logfwd_before_obs.delete(save=False)
