import { Component, OnInit } from '@angular/core';
import { Subject, from, merge, Observable } from "rxjs";
import { switchMap, map, windowCount, scan, take, tap } from "rxjs/operators";


import {
  ChatModule,
  Message,
  User,
  Action,
  ExecuteActionEvent,
  SendMessageEvent,
} from "@progress/kendo-angular-conversational-ui";
import { ChatService } from 'src/app/services/chat.service';
import { ChatbotModel } from 'src/app/model/chatbot-model';
import { HttpClient } from '@angular/common/http';
import { ChatbotMessageService } from 'src/app/services/chatbot-message.service';
import { FileRestrictions, RemoveEvent, SelectEvent } from '@progress/kendo-angular-upload';

@Component({
  selector: 'app-chatbot',
  templateUrl:'./chatbot.component.html',
  styleUrls: ['./chatbot.component.css'],
  providers: [ChatService],
  template: `
    <kendo-chat
      [messages]="feed | async"
      [user]="user"
      (sendMessage)="sendMessage($event)"
    >
    </kendo-chat>
  `
})
export class ChatbotComponent implements OnInit {


  ngOnInit(): void {
  }

  public feed: Observable<Message[]>;
  //public feed1: String;

  public readonly user: User = {
    id: 1,
  };

  public readonly bot: User = {
    id: 0,
  };

  private local: Subject<Message> = new Subject<Message>();

  constructor(private svc: ChatService,private httpClient: HttpClient, private chatbotService: ChatbotMessageService) {
    const hello: Message = {
      author: this.bot,
      suggestedActions: [
        {
          type: "reply",
          value: "Job Recomandation  ",
        },
        {
          type: "reply",
          value: "Advices and general discussion",
        },
      ],
      timestamp: new Date(),
      text: "Hello there how can I help you ?",
    };

    // Merge local and remote messages into a single stream
    this.feed = merge(
      from([hello]),
      this.local,
      this.chatbotService.responsess.pipe(
        map(
          (response): Message => ({
            author: this.bot,
            text: response,
          })
        )
      )
    ).pipe(
      // ... and emit an array of all messages
      scan((acc: Message[], x: Message) => [...acc, x], [])
    );
    

    
  }

  onButtonClick(){}

  public sendMessage(e: SendMessageEvent): void {
     console.log(e.message);
    let body= new ChatbotModel();
    let reponse="";
    body.the_question=e.message.text;
    console.log(body);
    this.chatbotService.addMessagee(body).subscribe(
      res=>{
        console.log(res);
        reponse=res.response;
        this.local.next(e.message);

        this.local.next({
          author: this.bot,
          typing: true,
        });
        console.log(e.message.text);
        console.log(reponse);
        this.chatbotService.submit(reponse);
      },
      err=>console.log(err)
    ) 
     
    //this.chatbotService.submit(reponse);
    //console.log(e.message) 
  }
  public events: string[] = [];
  public imagePreviews: any[] = [];

  public fileRestrictions: FileRestrictions = {
    allowedExtensions: [".jpg", ".png"],
  };

  public removeEventHandler(e: RemoveEvent): void {
    this.log(`Removing ${e.files[0].name}`);

    const index = this.imagePreviews.findIndex(
      (item) => item.uid === e.files[0].uid
    );

    if (index >= 0) {
      this.imagePreviews.splice(index, 1);
    }
  }

  public selectEventHandler(e: SelectEvent): void {
    const that = this;

    e.files.forEach((file) => {
      that.log(`File selected: ${file.name}`);
      let msg = `${file.name}`;
      let body= new ChatbotModel();
      let reponse = 'D:/pfe project/pfe project/' + msg; 
      body.the_question=reponse;
      console.log(body);
      this.chatbotService.addMessagee(body).subscribe(
        res=>{
          console.log(res);
          reponse=res.response;
          //this.local.next(e.message);

          this.local.next({
            author: this.bot,
            typing: true,
          });
         // console.log(e.message.text);
          console.log(reponse);
          this.chatbotService.submit(reponse);
        },
        err=>console.log(err)
      ) 
      if (!file.validationErrors) {
        const reader = new FileReader();

        reader.onload = function (ev) {
          const image = {
            src: ev.target["result"],
            uid: file.uid,
          };

          that.imagePreviews.unshift(image);
        };

        reader.readAsDataURL(file.rawFile);
      }
    });
  }

  private log(event: string): void {
    this.events.unshift(`${event}`);
  }

}
