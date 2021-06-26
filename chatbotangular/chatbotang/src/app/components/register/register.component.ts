import { Component, OnInit } from '@angular/core';
import { FirebaseService } from 'src/app/services/firebase.service';
import { Router } from '@angular/router';
@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent implements OnInit {

  isSignedIn=false

  constructor(public firebaseService:FirebaseService,private router:Router) { }

  ngOnInit(): void {
    if (localStorage.getItem('user')!==null)
    this.isSignedIn = true
    else 
    this.isSignedIn= false
  }
    onSingnup(email:string, password:string){
    console.log("aaa");
    this.firebaseService.signup(email,password)
    if (this.firebaseService.isLoggedIn){
    this.isSignedIn=false;
    this.router.navigate([''])
    }

  }
  login(){
    this.router.navigate(['']);
  }
}

