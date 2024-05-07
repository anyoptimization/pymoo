
const { createApp, reactive, ref } = Vue;

createApp({
  // Data 
  data(){
    return {
      imageWidgets : { }, 
      tableWidgets : { },
      highlightedWidget : ""
    }
  }, 
  // Created hook 
  created(){

    const evtSource = new EventSource("listen");
    
    evtSource.onmessage = (event) => {
    
      let data = JSON.parse(event.data)
    
      let title = data.title
    
      if(title == "Overview"){
        this.tableWidgets["Overview"] = data.content
      } else {
        this.imageWidgets[title] = data.content
      }

    };

  },
  // Methods
  methods : {
    slugify(str){
     return str.toLowerCase().trim().replace(/[^\w\s-]/g, '').replace(/[\s_-]+/g, '-').replace(/^-+|-+$/g, '');
    }, 
    highlightWidget(title){
      this.highlightedWidget = title;
    },
    widgetIsHighlighted(title){
      return title === this.highlightedWidget
    }

  }

  

}).mount('#app');

