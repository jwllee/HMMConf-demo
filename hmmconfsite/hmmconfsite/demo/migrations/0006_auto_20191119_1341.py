# Generated by Django 2.2.7 on 2019-11-19 13:41

from django.db import migrations
import hmmconfsite.demo.models
import sorl.thumbnail.fields


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0005_auto_20191116_1314'),
    ]

    operations = [
        migrations.AddField(
            model_name='log',
            name='file_barplot_logstartprob',
            field=sorl.thumbnail.fields.ImageField(blank=True, max_length=500, upload_to=hmmconfsite.demo.models.get_file_barplot_logstartprob_fp),
        ),
        migrations.AddField(
            model_name='log',
            name='file_net_logstartprob',
            field=sorl.thumbnail.fields.ImageField(blank=True, max_length=500, upload_to=hmmconfsite.demo.models.get_file_net_logstartprob_fp),
        ),
    ]
