import secrets
import string

from django.http import HttpResponse


def create_session_cookie(request):
    try:
        print(request.session['member_id'])
    except KeyError:
        request.session['member_id'] = __create_rand_alpha_num(8)

    return HttpResponse("Anonymous cookie created")


def __create_rand_alpha_num(n):
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for i in range(n))

