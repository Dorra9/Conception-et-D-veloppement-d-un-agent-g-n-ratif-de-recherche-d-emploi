import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';

import { HttpClientModule } from "@angular/common/http";

import { BrowserAnimationsModule } from "@angular/platform-browser/animations";
import { ChatModule } from "@progress/kendo-angular-conversational-ui";
import { UploadsModule } from "@progress/kendo-angular-upload";
import { ChatbotComponent } from './components/chatbot/chatbot.component';
import { AngularFireModule } from '@angular/fire';
import { CommonModule } from '@angular/common';
import { HomeComponent } from './components/home/home.component';
import { RegisterComponent } from './components/register/register.component';
import { SkillsComponent } from './components/skills/skills.component';
import { LangComponent } from './components/lang/lang.component';
import { JobsComponent } from './components/jobs/jobs.component';
import { CVComponent } from './components/cv/cv.component';
import { ProfileComponent } from './components/profile/profile.component';

@NgModule({
  declarations: [
    AppComponent,
    ChatbotComponent,
    HomeComponent,
    RegisterComponent,
    SkillsComponent,
    LangComponent,
    JobsComponent,
    CVComponent,
    ProfileComponent
  ],
  imports: [
    AngularFireModule.initializeApp({
      
        apiKey: "AIzaSyDTYlY9JFwWYsiA_-eyJqNGI7oy93kTZIs",
        authDomain: "chatbotangular-c9929.firebaseapp.com",
        projectId: "chatbotangular-c9929",
        storageBucket: "chatbotangular-c9929.appspot.com",
        messagingSenderId: "636474695481",
        appId: "1:636474695481:web:250f0093b2fe8077e3d2e8"
      

    }),
    BrowserModule,
    AppRoutingModule,
    ChatModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    CommonModule,
    AngularFireModule,
    UploadsModule
    

  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
