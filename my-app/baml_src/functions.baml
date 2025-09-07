class HumanMessage {
  message_type "request_clarification" | "respond_to_user"
  message string
}

enum Category {
  @@dynamic
}

function PickBestCategory(text: string) -> Category | HumanMessage {
    client CustomGeminiFlash
    prompt #"
        {{ _.role('system') }}
        Which category best describes the following text?

        {{ ctx.output_format }}

        {{ _.role('user') }}
        {{ text }}
    "#
}

test TestName {
  functions [PickBestCategory]
  type_builder {
    dynamic enum Category {
      Category1 @alias(k0) @description(#"
        for placeholder
      "#)
      Category2 @alias(k1) @description(#"
        for debugging
      "#)
      Category3 @alias(k2) @description(#"
        for general use
      "#) 
    }
  }
  args {
    text #"
      debug
    "#
  }
}





