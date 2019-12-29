from django.shortcuts import render

# Create your views here.

def index(request):
    from .models import User
    users = User.objects.all();
    p = users[len(users)-1].pic
    return render(request,'index.html', {'users':users})

def uploadImage(request):
    picture = request.FILES['image'];

    from .models import User
    user = User(pic=picture);
    user.save();

    return render(request,'index.html')
