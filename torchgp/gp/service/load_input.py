from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas
from django.core.cache import cache


@csrf_exempt
def load_input_in_cache(request):
    if request.method == 'POST':
        __handle_uploaded_file(request.FILES['file'], request.session['member_id'])
        return HttpResponse("Upload successful")
    else:
        return HttpResponse("Upload successful")


def __handle_uploaded_file(file, member_id):
    inp = pandas.read_csv(file)
    cache.set(member_id, inp, timeout=None)


@csrf_exempt
def peek_input(request):
    try:
        data = cache.get(request.session['member_id'])
        return HttpResponse(data.head().to_html())
    except:
        return HttpResponse("Unable to peek input data, upload again")
