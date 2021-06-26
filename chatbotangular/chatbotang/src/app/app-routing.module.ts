import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { ChatbotComponent } from './components/chatbot/chatbot.component';
import {HomeComponent} from './components/home/home.component';
import{RegisterComponent}from './components/register/register.component'
import { AcceuilComponent } from './components/acceuil/acceuil.component';
import { SkillsComponent } from './components/skills/skills.component';
import { LangComponent } from './components/lang/lang.component';
import { JobsComponent } from './components/jobs/jobs.component';
import { CVComponent } from './components/cv/cv.component';
import { ProfileComponent } from './components/profile/profile.component';

const routes: Routes = [
  {
    path: 'chatbot',
    component: ChatbotComponent
  },
  {
    path:'',
    component: HomeComponent

  },
  {
    path:'register',
    component: RegisterComponent

  },
  {
    path:'acceuil',
    component: AcceuilComponent
  },
  {
    path:'skills',
    component: SkillsComponent
  },
  {
    path:'lang',
    component: LangComponent
  },
  {
    path:'jobs',
    component: JobsComponent
  },
  {
    path:'cv',
    component: CVComponent
  },
  {
    path:'profile',
    component: ProfileComponent
  }


];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
