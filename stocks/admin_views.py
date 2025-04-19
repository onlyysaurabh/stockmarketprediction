import logging
from datetime import date, timedelta, datetime
from django.shortcuts import redirect
from django.contrib import messages
from django.http import HttpRequest, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admin.models import LogEntry, ADDITION
from django.contrib.contenttypes.models import ContentType
import json

from .models import Stock, StockNews, TrainedPredictionModel
from .forms import FetchNewsForm, TrainModelForm
from .prediction_models import xgboost_predictor, lstm_predictor, sarima_predictor, svm_predictor
from .prediction_data_service import get_training_data

logger = logging.getLogger(__name__)

# Your other admin views here...
