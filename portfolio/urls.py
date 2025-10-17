from django.urls import path
from .views import (
    ProjectList, ProjectDetail,
    AboutMeDetail,
    SkillList, SkillDetail,
    ExperienceList, ExperienceDetail,
    EducationList, EducationDetail,
    ChatView
)

urlpatterns = [
    path('projects/', ProjectList.as_view(), name='project-list'),
    path('projects/<int:pk>/', ProjectDetail.as_view(), name='project-detail'),

    path('about-me/', AboutMeDetail.as_view(), name='about-me-detail'),

    path('skills/', SkillList.as_view(), name='skill-list'),
    path('skills/<int:pk>/', SkillDetail.as_view(), name='skill-detail'),

    path('experience/', ExperienceList.as_view(), name='experience-list'),
    path('experience/<int:pk>/', ExperienceDetail.as_view(), name='experience-detail'),

    path('education/', EducationList.as_view(), name='education-list'),
    path('education/<int:pk>/', EducationDetail.as_view(), name='education-detail'),

    path('chat/', ChatView.as_view(), name='chat'),
]