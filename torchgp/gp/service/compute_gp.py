from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from ..model.GP import *
from django.core.cache import cache


@csrf_exempt
def best_effort_gp(request):
    gp_select = GPSelect(components=1).get_gp()
    gp = GP(gp_select, cache.get(request.session['member_id']))
    gp.fit()
    gp.plot_posterior()
    return HttpResponse("best effort plot successful")


@csrf_exempt
def plot_data(request):
    gp_select = GPSelect(components=1).get_gp()
    gp = GP(gp_select, cache.get(request.session['member_id']))
    gp.fit()
    gp.plot_data()
    return HttpResponse("Prior plot successful")
