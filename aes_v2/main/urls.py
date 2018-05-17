from django.conf.urls import url
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
   url(r'^$', views.home, name='home'),
   url(r'^login/$', auth_views.login, {'template_name':'main/login.html'}, name='login'),
   url(r'^logout/$', auth_views.logout, {'next_page':'/'}, name='logout'),
   url(r'^signup/$', views.signup, name='signup'),
   url(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$',
        views.activate, name='activate'),
   url(r'^scoring/$',views.get_essay,name='get_essay'),
]
