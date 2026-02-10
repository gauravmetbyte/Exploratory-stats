# =============================================
# ML + Optimization Shiny Platform (Minitab-like)
# Supports: GA, Neural Network, Random Forest, Bayesian Regression
# Upload Excel → Select Response & Predictors → Train → Optimize
# =============================================

library(shiny)
library(readxl)
library(randomForest)
library(neuralnet)
library(GA)
library(rstanarm)
library(ggplot2)
library(dplyr)

# ================= UI =================
ui <- fluidPage(
  titlePanel("ML Optimization Platform (RSM + AI)"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload Excel File"),
      uiOutput("varSelect"),
      selectInput("model", "Select Model",
                  c("Random Forest",
                    "Neural Network",
                    "Bayesian Regression",
                    "Genetic Algorithm (RSM)")),
      actionButton("train", "Train Model"),
      hr(),
      h4("Optimization"),
      actionButton("opt", "Run Optimization")
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("Data", tableOutput("dataView")),
        tabPanel("Model Summary", verbatimTextOutput("summary")),
        tabPanel("Prediction Plot", plotOutput("predPlot")),
        tabPanel("Optimization Result", tableOutput("optResult"))
      )
    )
  )
)

# ================= SERVER =================
server <- function(input, output, session) {

  # Load Data
  df <- reactive({
    req(input$file)
    read_excel(input$file$datapath)
  })

  # Variable Selection UI
  output$varSelect <- renderUI({
    req(df())

    vars <- names(df())

    tagList(
      selectInput("response", "Response Variable", vars),
      selectInput("predictors", "Predictors",
                  vars, multiple = TRUE)
    )
  })

  # Show Data
  output$dataView <- renderTable({
    req(df())
    head(df(), 20)
  })

  modelFit <- reactiveVal(NULL)

  # ================= TRAIN MODEL =================
  observeEvent(input$train, {

    req(df(), input$response, input$predictors)

    data <- df()
    y <- input$response
    x <- input$predictors

    formula <- as.formula(paste(y, "~", paste(x, collapse = "+")))

    # ---------------- RANDOM FOREST ----------------
    if (input$model == "Random Forest") {

      fit <- randomForest(formula, data = data)
      modelFit(fit)

    }

    # ---------------- NEURAL NETWORK ----------------
    if (input$model == "Neural Network") {

      scaled <- scale(data[, c(y, x)])
      scaled <- as.data.frame(scaled)

      f2 <- as.formula(paste(y, "~", paste(x, collapse = "+")))

      fit <- neuralnet(f2,
                       data = scaled,
                       hidden = c(5,3),
                       linear.output = TRUE)

      modelFit(list(model = fit,
                    scale = attr(scaled, "scaled:scale"),
                    center = attr(scaled, "scaled:center")))
    }

    # ---------------- BAYESIAN ----------------
    if (input$model == "Bayesian Regression") {

      fit <- stan_glm(formula,
                      data = data,
                      family = gaussian(),
                      chains = 2,
                      iter = 1000)

      modelFit(fit)
    }

    # ---------------- GA RSM ----------------
    if (input$model == "Genetic Algorithm (RSM)") {

      lmFit <- lm(formula, data = data)
      modelFit(lmFit)
    }

  })

  # ================= SUMMARY =================
  output$summary <- renderPrint({
    req(modelFit())

    if (input$model == "Neural Network") {
      print(modelFit()$model)
    } else {
      summary(modelFit())
    }
  })


  # ================= PREDICTION PLOT =================
  output$predPlot <- renderPlot({

    req(modelFit(), df())

    data <- df()
    y <- input$response

    pred <- NULL

    # RF
    if (input$model == "Random Forest") {
      pred <- predict(modelFit(), data)
    }

    # NN
    if (input$model == "Neural Network") {

      obj <- modelFit()
      scaled <- scale(data[, input$predictors],
                      center = obj$center[-1],
                      scale = obj$scale[-1])

      p <- compute(obj$model, scaled)
      pred <- p$net.result * obj$scale[1] + obj$center[1]
    }

    # Bayesian
    if (input$model == "Bayesian Regression") {
      pred <- predict(modelFit(), data)
    }

    # GA (LM)
    if (input$model == "Genetic Algorithm (RSM)") {
      pred <- predict(modelFit(), data)
    }


    plot(data[[y]], pred,
         xlab = "Actual",
         ylab = "Predicted",
         main = "Actual vs Predicted")
    abline(0,1,col="red")
  })


  # ================= OPTIMIZATION =================
  observeEvent(input$opt, {

    req(modelFit(), df())

    data <- df()
    x <- input$predictors

    mins <- apply(data[, x], 2, min)
    maxs <- apply(data[, x], 2, max)


    # Objective Function
    objFun <- function(par) {

      newdata <- as.data.frame(t(par))
      colnames(newdata) <- x

      if (input$model == "Random Forest") {
        -predict(modelFit(), newdata)
      }

      else if (input$model == "Bayesian Regression") {
        -predict(modelFit(), newdata)
      }

      else if (input$model == "Genetic Algorithm (RSM)") {
        -predict(modelFit(), newdata)
      }

      else {
        return(0)
      }

    }


    GAfit <- ga(type = "real-valued",
                fitness = objFun,
                min = mins,
                max = maxs,
                popSize = 50,
                maxiter = 100)


    best <- GAfit@solution
    colnames(best) <- x

    best <- as.data.frame(best)
    best$Predicted_Response <- -GAfit@fitnessValue

    output$optResult <- renderTable(best)

  })

}

# ================= RUN =================
shinyApp(ui, server)
