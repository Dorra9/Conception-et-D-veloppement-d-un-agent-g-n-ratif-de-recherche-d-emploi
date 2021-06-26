import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-profile',
  templateUrl: './profile.component.html',
  styleUrls: ['./profile.component.css']
})
export class ProfileComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  openDoc(pdfUrl: string, startPage: number ) {
    window.open(pdfUrl, '_blank', '', true);
  }

}
