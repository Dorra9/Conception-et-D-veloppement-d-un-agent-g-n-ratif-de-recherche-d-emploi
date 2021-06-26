import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ChatbotMessageService {

  constructor(private httpClient: HttpClient) { }
  public readonly responsess: Subject<string> = new Subject<string>();

  addMessagee(body){
    return this.httpClient.post<any>('http://127.0.0.1:5000/chatbot',body);
  }

  public submit(reponse: string) {
    //const length = question.length;
    console.log(reponse);
    const answer = `${reponse} `;
    console.log(answer);
    setTimeout(() => this.responsess.next(answer), 1000);
  }
}
