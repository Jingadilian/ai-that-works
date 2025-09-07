class HumanMessage {
  message_type "request_clarification" | "respond_to_user"
  message string
}

function PickAction(state: string) -> HumanMessage {
  client CustomGeminiFlash
  prompt #"
    You are an agent with access to any number of tools.

    {{ ctx.output_format }}

    Help the user by picking an action for the following.

    {{ _.role('user') }}
    {{ state }}
  "#
}

test HelloWorld {
  functions [PickAction]
  args {
    state #"
      Why is the sky blue?
    "#
  }
}



