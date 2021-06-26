import { Component, OnInit } from '@angular/core';
import { FirebaseService } from 'src/app/services/firebase.service';
import { Router } from '@angular/router';
import { AngularFireAuth } from '@angular/fire/auth';

import { FormGroup,FormControl,FormBuilder,Validators} from '@angular/forms';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
 
  loginForm:FormGroup;

  constructor(private fb :FormBuilder ,public auth:AngularFireAuth,private router:Router){
    let formControls ={
      email:new FormControl('',[
        
        Validators.email
      ]),
      password:new FormControl('',[
      
      Validators.minLength(8)
      ])
    }
    this.loginForm =fb.group(formControls);
  }
  get email(){return this.loginForm.get('email');}
  get password(){return this.loginForm.get('password');}

  ngOnInit(): void { 
  }
  login(){
    let data=this.loginForm.value;
    console.log(this.loginForm.value);
    this.auth.signInWithEmailAndPassword(data.email,data.password).then(res=>{console.log(res);
      this.router.navigate(['/acceuil'])

    }); 
    
  }

  register(){
    this.router.navigate(['/register']);
  }

}
